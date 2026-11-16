//! Speech editing: a recording's words changed - the regions the transcript diff touches are said anew in the
//! recording's voice by the speech-edit backbone (the edit build's LM with its LoRA), everything else kept codec frame
//! for codec frame, and the whole decoded once.
//!
//! ```text
//!   recording (<= 30 s) + its transcript -> FuseAlign word timings (align.rs) -> edit tokens (textmatch)
//!   difflib over the tokens against the target's -> regions (frames, 0.08 s margins, at most 3)
//!   the recording's codes (the reference encoder) and speaker embedding (MECT)
//!   per window (the whole clip up to 16 s, else words within 4 s of the regions) and region:
//!     prompt [speaker][conditions] BOS target EOS, the clip's codes with a marker at each region, earlier fills, marker
//!     -> the edit LM samples until span-end (masked until 70 % of the expected length), EOA never
//!   the fills spliced into the clip's codes -> one decode
//! ```
//!
//! Origin: Zyphra's ZONOS3-Mini runtime (model/src/zonos3/tts/edit.py, edit_align.py, edit_runtime.py, textmatch.py,
//! align_text.py and their in-browser port servers/web/src/edit, api/edit.ts; MIT). Not ported: CJK word segmentation of
//! unspaced languages (their words are cut at spaces here), caller-supplied word timings, region controls, verify, and
//! clips over FuseAlign's 30 s.

use nextsycl_audio::{Error, Result};

/// A region: (start frame, length in frames)
pub type Span = (usize, usize);

pub const FRAME_RATE: f64 = 15.0;

// ------------------------------------------------------------------------------------------------------------ text

/// textmatch.normalize: curly quotes as ', '_' a space, upper-cased, everything but letters, digits and ' one space
pub fn normalize(text: &str) -> String {
    let mut out = String::new();
    let mut space = false;
    for ch in text.chars() {
        let ch = match ch {
            '\u{2018}' | '\u{2019}' | '\u{201b}' | '\u{02bc}' => '\'',
            '_' => ' ',
            c => c,
        };
        if ch.is_alphanumeric() || ch == '\'' {
            if space && !out.is_empty() {
                out.push(' ');
            }
            space = false;
            out.extend(ch.to_uppercase());
        } else {
            space = true;
        }
    }
    out
}

/// textmatch.tokens: the edit tokens of a text
pub fn tokens(text: &str) -> Vec<String> {
    normalize(text).split_whitespace().map(str::to_string).collect()
}

/// The words FuseAlign places: the whitespace words
pub fn align_tokens(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

/// (all target tokens, each target word's tokens)
pub fn target_tokens(target: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let words: Vec<Vec<String>> = target.split_whitespace().map(tokens).collect();
    (words.iter().flatten().cloned().collect(), words)
}

// ----------------------------------------------------------------------------------------------------------- words

/// An aligned word of the original with its edit tokens
#[derive(Clone, Debug)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f32,
    pub tokens: Vec<String>,
}

/// aligned_words: tokenless words dropped, an overlap with the word before split at its midpoint
pub fn aligned_words(words: &[(String, f64, f64, f32)]) -> Vec<Word> {
    let mut out: Vec<Word> = words.iter().filter_map(|(w, s, e, p)| {
        let t = tokens(w);
        (!t.is_empty()).then(|| Word { word: w.clone(), start: *s, end: *e, confidence: *p, tokens: t })
    }).collect();
    for k in 1..out.len() {
        if out[k].start < out[k - 1].end {
            let m = (out[k - 1].end + out[k].start) / 2.0;
            out[k - 1].end = m;
            out[k].start = m;
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    Equal,
    Replace,
    Delete,
    Insert,
}

pub type Opcode = (Tag, usize, usize, usize, usize);

/// difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
pub fn opcodes(a: &[String], b: &[String]) -> Vec<Opcode> {
    use std::collections::HashMap;
    let mut b2j: HashMap<&str, Vec<usize>> = HashMap::new();
    for (j, t) in b.iter().enumerate() {
        b2j.entry(t.as_str()).or_default().push(j);
    }
    let longest = |alo: usize, ahi: usize, blo: usize, bhi: usize| -> (usize, usize, usize) {
        let (mut bi, mut bj, mut bs) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut next = HashMap::new();
            if let Some(js) = b2j.get(a[i].as_str()) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = j.checked_sub(1).and_then(|p| j2len.get(&p)).copied().unwrap_or(0) + 1;
                    next.insert(j, k);
                    if k > bs {
                        bi = i + 1 - k;
                        bj = j + 1 - k;
                        bs = k;
                    }
                }
            }
            j2len = next;
        }
        (bi, bj, bs)
    };
    let mut found = Vec::new();
    let mut queue = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest(alo, ahi, blo, bhi);
        if k == 0 {
            continue;
        }
        found.push((i, j, k));
        if alo < i && blo < j {
            queue.push((alo, i, blo, j));
        }
        if i + k < ahi && j + k < bhi {
            queue.push((i + k, ahi, j + k, bhi));
        }
    }
    found.sort();
    let mut blocks = Vec::new();
    let (mut i1, mut j1, mut k1) = (0, 0, 0);
    for (i2, j2, k2) in found {
        if i1 + k1 == i2 && j1 + k1 == j2 {
            k1 += k2;
        } else {
            if k1 > 0 {
                blocks.push((i1, j1, k1));
            }
            (i1, j1, k1) = (i2, j2, k2);
        }
    }
    if k1 > 0 {
        blocks.push((i1, j1, k1));
    }
    blocks.push((a.len(), b.len(), 0));
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    for (ai, bj, size) in blocks {
        let tag = if i < ai && j < bj { Some(Tag::Replace) } else if i < ai { Some(Tag::Delete) } else if j < bj { Some(Tag::Insert) } else { None };
        if let Some(t) = tag {
            out.push((t, i, ai, j, bj));
        }
        i = ai + size;
        j = bj + size;
        if size > 0 {
            out.push((Tag::Equal, ai, i, bj, j));
        }
    }
    out
}

/// diff_spans: the frame regions of the original words the target's tokens change, each widened by `margin` seconds;
/// regions closer than 5 frames merge
pub fn diff_spans(words: &[Word], target: &[String], total: usize, margin: f64, max_spans: usize) -> Result<Vec<Span>> {
    let expanded: Vec<(&String, &Word)> = words.iter().flat_map(|w| w.tokens.iter().map(move |t| (t, w))).collect();
    let toks: Vec<String> = expanded.iter().map(|(t, _)| (*t).clone()).collect();
    let mut spans: Vec<Span> = Vec::new();
    for (op, i1, i2, _, _) in opcodes(&toks, target) {
        if op == Tag::Equal {
            continue;
        }
        let (t0, t1) = if i1 != i2 {
            (expanded[i1].1.start, expanded[i2 - 1].1.end)
        } else {
            (if i1 > 0 { expanded[i1 - 1].1.end } else { 0.0 }, if i1 < expanded.len() { expanded[i1].1.start } else { total as f64 / FRAME_RATE })
        };
        let start = (((t0 - margin) * FRAME_RATE).floor().max(0.0) as usize).min(total.saturating_sub(1));
        let end = (((t1 + margin) * FRAME_RATE).ceil() as usize).min(total).max(start + 1);
        match spans.last_mut() {
            Some(last) if (start as i64) - ((last.0 + last.1) as i64) < 5 => last.1 = end.max(last.0 + last.1) - last.0,
            _ => spans.push((start, end - start)),
        }
    }
    if spans.is_empty() {
        return Err(Error("the target text is the recording's own transcript: nothing to change".into()));
    }
    if spans.len() > max_spans {
        return Err(Error(format!("the edit touches {} regions; at most {max_spans} are supported", spans.len())));
    }
    Ok(spans)
}

/// The original's words against the target's, diffed over their tokens: original word ranges onto target word ranges
pub struct WordDiff {
    orig_start: Vec<usize>,
    tgt_start: Vec<usize>,
    ops: Vec<Opcode>,
    n: usize,
}

impl WordDiff {
    pub fn new(words: &[Word], target_word_tokens: &[Vec<String>]) -> WordDiff {
        let mut orig_start = vec![0];
        for w in words {
            orig_start.push(orig_start.last().unwrap() + w.tokens.len());
        }
        let mut tgt_start = vec![0];
        for t in target_word_tokens {
            tgt_start.push(tgt_start.last().unwrap() + t.len());
        }
        let a: Vec<String> = words.iter().flat_map(|w| w.tokens.clone()).collect();
        let b: Vec<String> = target_word_tokens.iter().flatten().cloned().collect();
        WordDiff { orig_start, tgt_start, ops: opcodes(&a, &b), n: words.len() }
    }

    fn target_cut(&self, i: usize, left: bool) -> Option<usize> {
        let ops: Vec<&Opcode> = if left { self.ops.iter().collect() } else { self.ops.iter().rev().collect() };
        for &&(tag, i1, i2, j1, j2) in &ops {
            if i1 < i && i < i2 {
                return (tag == Tag::Equal).then(|| j1 + i - i1);
            }
            if (if left { i1 } else { i2 }) == i {
                return Some(if left { j1 } else { j2 });
            }
        }
        Some(if i == 0 { 0 } else { *self.tgt_start.last().unwrap() })
    }

    fn target_word(&self, k: usize, left: bool) -> Option<usize> {
        let j = self.target_cut(self.orig_start[k], left)?;
        let at: Vec<usize> = self.tgt_start.iter().enumerate().filter(|(_, s)| **s == j).map(|(t, _)| t).collect();
        if at.is_empty() {
            return None;
        }
        Some(if left { at[0] } else { *at.last().unwrap() })
    }

    pub fn clean(&self, mut a: usize, mut b: usize) -> (usize, usize) {
        while a > 0 && self.target_word(a, true).is_none() {
            a -= 1;
        }
        while b < self.n && self.target_word(b, false).is_none() {
            b += 1;
        }
        (a, b)
    }

    pub fn target_range(&self, a: usize, b: usize) -> (Option<usize>, Option<usize>) {
        (self.target_word(a, true), self.target_word(b, false))
    }
}

fn slice<T: Clone>(v: &[T], a: Option<usize>, b: Option<usize>) -> Vec<T> {
    let (a, b) = (a.unwrap_or(0), b.unwrap_or(v.len()));
    if a >= b { Vec::new() } else { v[a..b.min(v.len())].to_vec() }
}

/// The audio frames [start, end) the model sees for some regions, and the target text of that stretch
#[derive(Clone, Debug)]
pub struct Window {
    pub start: usize,
    pub end: usize,
    pub regions: Vec<usize>,
    pub text: String,
}

/// plan_edit_windows: words within `context` seconds of each region, edges moved out of changed blocks, overlapping
/// windows merged, cut halfway between their edge words and the neighbours
pub fn plan_windows(spans: &[Span], words: &[Word], target_words: &[String], target_word_tokens: &[Vec<String>], total: usize, context: f64,
                    target_text: &str) -> Result<Vec<Window>> {
    let n = words.len();
    if n == 0 {
        return Err(Error("the aligner found no words in the recording, so the edit cannot be windowed".into()));
    }
    let diff = WordDiff::new(words, target_word_tokens);
    let boundary = |k: usize| -> usize {
        if k == 0 {
            0
        } else if k == n {
            total
        } else {
            (((words[k - 1].end + words[k].start) / 2.0 * FRAME_RATE).round_ties_even() as usize).min(total)
        }
    };
    let mut wins: Vec<(usize, usize, Vec<usize>)> = spans.iter().enumerate().map(|(index, (start, length))| {
        let t0 = *start as f64 / FRAME_RATE - context;
        let t1 = (start + length) as f64 / FRAME_RATE + context;
        let a = words.iter().position(|w| w.end > t0).unwrap_or(n);
        let b = (0..n).rev().find(|k| words[*k].start < t1).map_or(0, |k| k + 1);
        let (ca, cb) = diff.clean(a, a.max(b));
        (ca, cb, vec![index])
    }).collect();
    let frames = |w: &(usize, usize, Vec<usize>)| -> (usize, usize) {
        (w.2.iter().map(|k| spans[*k].0).fold(boundary(w.0), usize::min), w.2.iter().map(|k| spans[*k].0 + spans[*k].1).fold(boundary(w.1), usize::max))
    };
    wins.sort_by_key(|x| (x.0, x.1));
    let mut merged = vec![wins[0].clone()];
    for w in wins.into_iter().skip(1) {
        let cur = merged.last_mut().unwrap();
        if w.0 < cur.1 || frames(&w).0 < frames(cur).1 {
            cur.0 = cur.0.min(w.0);
            cur.1 = cur.1.max(w.1);
            cur.2.extend(w.2);
            cur.2.sort();
        } else {
            merged.push(w);
        }
    }
    Ok(merged.iter().map(|w| {
        let (start, end) = frames(w);
        let text = if w.0 == 0 && w.1 == n {
            target_text.to_string()
        } else {
            let (ja, jb) = diff.target_range(w.0, w.1);
            slice(target_words, ja, jb).join(" ")
        };
        Window { start, end, regions: w.2.clone(), text }
    }).collect())
}

/// expected_fill_seconds: each region's seconds scaled by the target over the original characters (edit tokens) of
/// its words; a region holding no original word takes the clip's characters a second
pub fn expected_seconds(spans: &[Span], words: &[Word], target_word_tokens: &[Vec<String>]) -> Vec<f64> {
    let diff = WordDiff::new(words, target_word_tokens);
    let chars = |lists: &[Vec<String>]| -> f64 { lists.iter().flatten().map(|t| t.chars().count() as f64).sum() };
    let spoken: f64 = words.iter().map(|w| w.end - w.start).sum();
    let all: Vec<Vec<String>> = words.iter().map(|w| w.tokens.clone()).collect();
    let rate = if spoken > 0.0 { chars(&all) / spoken } else { 0.0 };
    let mids: Vec<f64> = words.iter().map(|w| (w.start + w.end) / 2.0).collect();
    spans.iter().map(|(start, length)| {
        let (r0, r1) = (*start as f64 / FRAME_RATE, (start + length) as f64 / FRAME_RATE);
        let inside: Vec<usize> = mids.iter().enumerate().filter(|(_, m)| r0 <= **m && **m < r1).map(|(k, _)| k).collect();
        let a = inside.first().copied().unwrap_or_else(|| mids.iter().position(|m| *m >= r0).unwrap_or(words.len()));
        let b = inside.last().map_or(a, |l| l + 1);
        let (ca, cb) = diff.clean(a, b);
        let original = chars(&all[ca..cb]);
        let (ja, jb) = diff.target_range(ca, cb);
        let target = chars(&slice(target_word_tokens, ja, jb));
        if original > 0.0 {
            (r1 - r0) * target / original
        } else if rate > 0.0 {
            target / rate
        } else {
            r1 - r0
        }
    }).collect()
}

// ---------------------------------------------------------------------------------------------------------- frames

pub fn validate_spans(spans: &[Span], total: usize, max: usize) -> Result<()> {
    if spans.is_empty() || spans.len() > max {
        return Err(Error("an edit takes between one and max_spans regions".into()));
    }
    let mut prev: Option<usize> = None;
    for (s, l) in spans {
        if *l < 1 || s + l > total || prev.is_some_and(|p| *s <= p) {
            return Err(Error("edit regions must be ascending, separated and within the audio".into()));
        }
        prev = Some(s + l);
    }
    Ok(())
}

/// sheared_block: the delay-stacked frames (+ one all-`end` row) then ncb - 1 pad rows, text column the text pad
fn sheared(frames: &[[u32; 10]], ncb: usize, pad: u32, text_pad: u32, end: Option<u32>) -> Vec<u32> {
    let n = frames.len();
    let rows = n + end.is_some() as usize + ncb - 1;
    let mut out = Vec::with_capacity(rows * (ncb + 1));
    for r in 0..rows {
        for c in 0..ncb {
            let v = match r.checked_sub(c) {
                Some(f) if f < n => frames[f][c],
                Some(f) if f == n && end.is_some() => end.unwrap(),
                _ => pad,
            };
            out.push(v);
        }
        out.push(text_pad);
    }
    out
}

/// edit_prompt_tokens: the text rows, the unmasked context (each region its marker, ended by EOA), every earlier fill
/// after its marker (ended by span-end), then the next region's marker
#[allow(clippy::too_many_arguments)]
pub fn edit_prompt(text_rows: &[u32], codes: &[[u32; 10]], spans: &[Span], fills: &[Vec<[u32; 10]>], ncb: usize, pad: u32, text_vocab: u32, eoa: u32,
                   span_end: u32) -> Result<Vec<u32>> {
    validate_spans(spans, codes.len(), spans.len())?;
    if fills.len() >= spans.len() {
        return Err(Error("the edit prompt needs a region not yet generated".into()));
    }
    let marker = |j: usize, out: &mut Vec<u32>| {
        out.extend(std::iter::repeat_n(pad, ncb));
        out.push(text_vocab + 1 + j as u32);
    };
    let mut out = text_rows.to_vec();
    let mut cursor = 0;
    for (j, (s, l)) in spans.iter().enumerate() {
        if *s > cursor {
            out.extend(sheared(&codes[cursor..*s], ncb, pad, text_vocab, None));
        }
        marker(j, &mut out);
        cursor = s + l;
    }
    out.extend(sheared(&codes[cursor..], ncb, pad, text_vocab, Some(eoa)));
    for (j, f) in fills.iter().enumerate() {
        marker(j, &mut out);
        out.extend(sheared(f, ncb, pad, text_vocab, Some(span_end)));
    }
    marker(fills.len(), &mut out);
    Ok(out)
}

/// extract_fill: the codec frames of a generated region (the rows without the finishing one) and whether it ended on
/// span-end; cut before the first frame holding a non-code
pub fn extract_fill(rows: &[[u32; 10]], eos: Option<usize>, ncb: usize, codebook: u32) -> (Vec<[u32; 10]>, bool) {
    let h = rows.len();
    let mut n = h.min(eos.unwrap_or((h + 1).saturating_sub(ncb)));
    let mut truncated = false;
    'f: for f in 0..n {
        for c in 0..ncb {
            let v = if f + c < h { rows[f + c][c] } else { codebook + 1 };
            if v >= codebook {
                truncated = true;
                n = f;
                break 'f;
            }
        }
    }
    let fill = (0..n).map(|f| {
        let mut fr = [0u32; 10];
        for (c, x) in fr.iter_mut().enumerate().take(ncb) {
            *x = rows[f + c][c];
        }
        fr
    }).collect();
    (fill, eos.is_some() && !truncated)
}

/// splice_edit: the clip's codes with each region replaced by its fill
pub fn splice(codes: &[[u32; 10]], spans: &[Span], fills: &[Vec<[u32; 10]>]) -> Vec<[u32; 10]> {
    let mut out = Vec::new();
    let mut cursor = 0;
    for ((s, l), f) in spans.iter().zip(fills) {
        out.extend_from_slice(&codes[cursor..*s]);
        out.extend_from_slice(f);
        cursor = s + l;
    }
    out.extend_from_slice(&codes[cursor..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn tokens_are_textmatchs() {
        assert_eq!(tokens("It’s the cat_dog, ok?"), s(&["IT'S", "THE", "CAT", "DOG", "OK"]));
        assert_eq!(tokens("Über 3 Äpfel"), s(&["ÜBER", "3", "ÄPFEL"]));
    }

    #[test]
    fn difflib_opcodes() {
        let a = s(&["THE", "QUICK", "BROWN", "FOX"]);
        let b = s(&["THE", "SLOW", "BROWN", "FOX", "JUMPS"]);
        assert_eq!(opcodes(&a, &b), vec![(Tag::Equal, 0, 1, 0, 1), (Tag::Replace, 1, 2, 1, 2), (Tag::Equal, 2, 4, 2, 4), (Tag::Insert, 4, 4, 4, 5)]);
    }

    #[test]
    fn a_changed_word_becomes_its_region() {
        let w = aligned_words(&[("The".into(), 0.1, 0.3, 1.0), ("quick".into(), 0.3, 0.7, 1.0), ("fox".into(), 0.8, 1.2, 1.0)]);
        let spans = diff_spans(&w, &s(&["THE", "SLOW", "FOX"]), 30, 0.08, 3).unwrap();
        assert_eq!(spans, vec![(3, 9)]);
        let (all, words) = target_tokens("The slow fox");
        assert_eq!(all, s(&["THE", "SLOW", "FOX"]));
        let e = expected_seconds(&spans, &w, &words);
        // "The" (its middle at 0.2 s) and "quick" fall in the region: 8 characters said in 7
        assert!((e[0] - 0.6 * 7.0 / 8.0).abs() < 1e-9, "{e:?}");
    }

    #[test]
    fn the_prompt_masks_the_region_and_the_fill_splices_back() {
        let codes: Vec<[u32; 10]> = (0..6).map(|f| [f as u32; 10]).collect();
        let p = edit_prompt(&[9; 11], &codes, &[(2, 2)], &[], 10, 4097, 8196, 4096, 4098).unwrap();
        // text row, 2 frames + 9 pad rows, the marker, 2 frames + EOA + 9, the next marker
        assert_eq!(p.len() / 11, 1 + 11 + 1 + 12 + 1);
        assert_eq!(p[12 * 11 + 10], 8197);
        let fill = vec![[7u32; 10]; 3];
        let sp = splice(&codes, &[(2, 2)], &[fill]);
        assert_eq!(sp.len(), 7);
        assert_eq!(sp[2], [7; 10]);
        assert_eq!(sp[5], [4; 10]);
    }
}
