//! Rule-based joint-name cleaning: a rig's bone names ("mixamorig:LeftUpLeg", "FR_calf", "Bip01 R Thigh") to the labels
//! UniMate was trained on - `[Left |Right ][Qualifier ]Part[ End]` ("Left Front Thigh") - which its text encoder turns
//! into the joint embeddings. A compact port of data_process/joint_annotation/names_clean_rule.py (prefix stripping,
//! side tokens, camel-case words, the canonical vocabulary of `vocab`); the reference's species-specific maps and its
//! LLM stage are not here. A name nothing in the vocabulary recognizes (`bone_0`, `Bone.012`) gives None: the caller
//! labels it "Bone" (or leaves it out: the structure-only mode).

use crate::vocab::{CANONICAL, FOREIGN_PARTS, JAPANESE_WORDS, MIXAMO_MAP, QUALIFIERS, REMOVE_PREFIXES, SIDE_WORDS, STANDALONE_MAP};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    pub fn word(self) -> &'static str {
        match self {
            Side::Left => "Left",
            Side::Right => "Right",
        }
    }
}

/// A recognized name
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    pub side: Option<Side>,
    /// "Front Thigh", "Head End": qualifier, part, end marker
    pub part: String,
}

impl Label {
    /// The training format: "Left Front Thigh"
    pub fn text(&self) -> String {
        match self.side {
            Some(s) => format!("{} {}", s.word(), self.part),
            None => self.part.clone(),
        }
    }
}

fn lookup(table: &'static [(&'static str, &'static str)], key: &str) -> Option<&'static str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// The words of a raw name: capital runs ("CAT", "R"), Capitalized or lower-case words, digit runs
fn words(s: &str) -> Vec<String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_ascii_digit() {
            let j = c[i..].iter().position(|x| !x.is_ascii_digit()).map_or(c.len(), |p| i + p);
            out.push(c[i..j].iter().collect());
            i = j;
        } else if ch.is_ascii_uppercase() {
            // a capital run not followed by a lower-case letter, else one Capitalized word
            let mut j = i;
            while j < c.len() && c[j].is_ascii_uppercase() {
                j += 1;
            }
            if j < c.len() && c[j].is_ascii_lowercase() {
                if j - i > 1 {
                    out.push(c[i..j - 1].iter().collect());
                    i = j - 1;
                }
                let mut k = i + 1;
                while k < c.len() && c[k].is_ascii_lowercase() {
                    k += 1;
                }
                out.push(c[i..k].iter().collect());
                i = k;
            } else {
                out.push(c[i..j].iter().collect());
                i = j;
            }
        } else if ch.is_ascii_lowercase() {
            let j = c[i..].iter().position(|x| !x.is_ascii_lowercase()).map_or(c.len(), |p| i + p);
            out.push(c[i..j].iter().collect());
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn capitalize(w: &str) -> String {
    let mut c = w.chars();
    match c.next() {
        Some(f) => f.to_ascii_uppercase().to_string() + &c.as_str().to_ascii_lowercase(),
        None => String::new(),
    }
}

/// A part word or two joined: the canonical label, "" for rig noise, None for an unknown word
fn part_of(word: &str) -> Option<String> {
    let cap = capitalize(word);
    let low = word.to_ascii_lowercase();
    for key in [word, cap.as_str()] {
        if let Some(v) = lookup(CANONICAL, key).or_else(|| lookup(MIXAMO_MAP, key)) {
            return Some(v.to_string());
        }
    }
    // a leg's hip joint on its own (legged rigs: FR_hip), not the pelvis "Hips"
    if low == "hip" {
        return Some("Hip".to_string());
    }
    lookup(FOREIGN_PARTS, &low).or_else(|| lookup(JAPANESE_WORDS, &low)).map(str::to_string)
}

/// Strips what rigs add around a name: parenthesised notes, FBX escapes, a namespace, known prefixes, Biped's "Bip01_"
fn strip_prefix(raw: &str) -> String {
    let mut s = raw.to_string();
    while let (Some(a), Some(b)) = (s.find('('), s.find(')')) {
        if a < b {
            s.replace_range(a..=b, "");
        } else {
            break;
        }
    }
    s = s.replace("FBXASC032", " ");
    if let Some(i) = s.rfind(':') {
        s = s[i + 1..].to_string();
    }
    for _ in 0..3 {
        for p in REMOVE_PREFIXES {
            if let Some(rest) = s.strip_prefix(p) {
                if !rest.is_empty() {
                    s = rest.to_string();
                }
            }
        }
    }
    // Bip01_ / Bip001 / Bip01-: any index
    if let Some(rest) = s.strip_prefix("Bip") {
        let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 {
            let tail = rest[digits..].trim_start_matches(['_', '-', ' ']);
            if !tail.is_empty() {
                s = tail.to_string();
            }
        }
    }
    s.trim().to_string()
}

/// The label of a bone name, or None when nothing in it is recognized
pub fn clean(raw: &str) -> Option<Label> {
    let mut s = strip_prefix(raw);
    if let Some(v) = lookup(STANDALONE_MAP, &s) {
        return Some(split_side(v));
    }
    let mut side = None;
    let mut end = false;
    // trailing decorations: ".001", "_037", ".L", "_r", ".x" in any order
    loop {
        let b = s.as_bytes();
        let n = b.len();
        if n >= 2 && (b[n - 2] == b'.' || b[n - 2] == b'_') && matches!(b[n - 1], b'L' | b'R' | b'l' | b'r' | b'x') {
            if side.is_none() {
                side = match b[n - 1] {
                    b'L' | b'l' => Some(Side::Left),
                    b'R' | b'r' => Some(Side::Right),
                    _ => None,
                };
            }
            s.truncate(n - 2);
            continue;
        }
        let digits = s.chars().rev().take_while(char::is_ascii_digit).count();
        if digits > 0 && digits < n && (b[n - digits - 1] == b'.' || b[n - digits - 1] == b'_') {
            s.truncate(n - digits - 1);
            continue;
        }
        break;
    }
    for suf in ["_end", ".end", "End", "END", "Nub", "_End"] {
        if s.len() > suf.len() && s.ends_with(suf) {
            s.truncate(s.len() - suf.len());
            end = true;
            break;
        }
    }
    let s = s.trim_matches(['_', '.', ' ', '-']).to_string();
    if let Some(v) = lookup(STANDALONE_MAP, &s) {
        let mut l = split_side(v);
        l.side = l.side.or(side);
        if end {
            l.part += " End";
        }
        return Some(l);
    }
    // legged robots and quadrupeds: FR_hip, RL_calf (front / back, then the side)
    let leg = {
        let b: Vec<char> = s.chars().collect();
        (b.len() > 3 && b[2] == '_' || b.len() > 3 && b[2] == '.') && "FRBHfrbh".contains(b[0]) && "LRlr".contains(b[1])
    };
    let mut qualifier = None;
    let mut body = s.clone();
    if leg {
        let c: Vec<char> = s.chars().collect();
        qualifier = Some(if matches!(c[0], 'F' | 'f') { "Front" } else { "Back" });
        side = side.or(Some(if matches!(c[1], 'L' | 'l') { Side::Left } else { Side::Right }));
        body = c[3..].iter().collect();
    }
    // a single leading L / R before a Capitalized word ("LThigh", "RUpLeg")
    {
        let c: Vec<char> = body.chars().collect();
        if c.len() > 2 && (c[0] == 'L' || c[0] == 'R') && c[1].is_ascii_uppercase() && c[2].is_ascii_lowercase() && side.is_none() {
            side = Some(if c[0] == 'L' { Side::Left } else { Side::Right });
            body = c[1..].iter().collect();
        }
    }
    // the whole body as one canonical key first (UpperArm, ForeArm, UpLeg, HeadTop_End)
    let joined: String = body.split(['_', ' ', '.', '-']).map(|w| w.to_string()).collect::<Vec<_>>().concat();
    let ws: Vec<String> = body.split(['_', ' ', '.', '-']).flat_map(words).collect();
    let mut parts: Vec<String> = Vec::new();
    let mut quals: Vec<&str> = qualifier.into_iter().collect();
    if let Some(v) = part_of(&joined).filter(|v| !v.is_empty()) {
        parts.push(v);
    } else {
        let mut i = 0;
        while i < ws.len() {
            let w = &ws[i];
            let low = w.to_ascii_lowercase();
            if w.chars().all(|c| c.is_ascii_digit()) {
                i += 1;
                continue;
            }
            if (w == "L" || w == "R" || w == "l" || w == "r") && ws.len() > 1 {
                if side.is_none() {
                    side = Some(if low == "l" { Side::Left } else { Side::Right });
                }
                i += 1;
                continue;
            }
            if let Some((_, v)) = SIDE_WORDS.iter().find(|(k, _)| *k == low) {
                if side.is_none() {
                    side = Some(if *v == "Left" { Side::Left } else { Side::Right });
                }
                i += 1;
                continue;
            }
            if let Some(q) = QUALIFIERS.iter().find(|q| q.eq_ignore_ascii_case(w)) {
                let q = if *q == "Rear" || *q == "Hind" { "Back" } else { q };
                if i + 1 < ws.len() {
                    quals.push(q);
                    i += 1;
                    continue;
                }
            }
            // two words as one key (Fore + Arm)
            if i + 1 < ws.len() {
                if let Some(v) = part_of(&format!("{w}{}", ws[i + 1])).filter(|v| !v.is_empty()) {
                    parts.push(v);
                    i += 2;
                    continue;
                }
            }
            if let Some(v) = part_of(w) {
                if !v.is_empty() {
                    parts.push(v);
                }
            }
            i += 1;
        }
    }
    let base = parts.last()?.clone();
    let mut part = String::new();
    let mut quals_seen: Vec<&str> = Vec::new();
    for q in quals {
        if !quals_seen.contains(&q) && !base.starts_with(q) {
            quals_seen.push(q);
        }
    }
    for q in quals_seen {
        part += q;
        part += " ";
    }
    part += &base;
    if end && !part.ends_with(" End") {
        part += " End";
    }
    let mut l = split_side(&part);
    l.side = l.side.or(side);
    Some(l)
}

/// "Left Thigh" -> (Left, "Thigh")
fn split_side(v: &str) -> Label {
    if let Some(r) = v.strip_prefix("Left ") {
        Label { side: Some(Side::Left), part: r.to_string() }
    } else if let Some(r) = v.strip_prefix("Right ") {
        Label { side: Some(Side::Right), part: r.to_string() }
    } else {
        Label { side: None, part: v.to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(raw: &str) -> Option<String> {
        clean(raw).map(|l| l.text())
    }

    #[test]
    fn mixamo_names() {
        assert_eq!(t("mixamorig:Hips").as_deref(), Some("Hips"));
        assert_eq!(t("mixamorig:LeftUpLeg").as_deref(), Some("Left Thigh"));
        assert_eq!(t("mixamorig:RightForeArm").as_deref(), Some("Right Forearm"));
        assert_eq!(t("mixamorig:HeadTop_End").as_deref(), Some("Head End"));
    }

    #[test]
    fn legged_and_decorated_names() {
        assert_eq!(t("FR_thigh").as_deref(), Some("Right Front Thigh"));
        assert_eq!(t("RL_calf").as_deref(), Some("Left Back Shin"));
        assert_eq!(t("Head_upper").as_deref(), Some("Head"));
        assert_eq!(t("Thigh.L").as_deref(), Some("Left Thigh"));
        assert_eq!(t("Bip01 R Thigh").as_deref(), Some("Right Thigh"));
        assert_eq!(t("Spine1").as_deref(), Some("Spine"));
        assert_eq!(t("LThigh").as_deref(), Some("Left Thigh"));
        assert_eq!(t("FR_hip").as_deref(), Some("Right Front Hip"));
    }

    #[test]
    fn generic_names_are_not_recognized() {
        assert_eq!(t("bone_0"), None);
        assert_eq!(t("Bone.012"), None);
        assert_eq!(t("joint16.001"), None);
    }
}
