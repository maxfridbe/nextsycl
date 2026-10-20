//! Word times from the decoder's cross-attention, as openai-whisper's timing.py makes them: the alignment heads'
//! attention over the window's frames (softmaxed, normalized across the tokens, median-filtered, averaged), a dynamic
//! time warping of the tokens against the frames, the words' first frames read off where the path moves to a new
//! token; then its punctuation merges and its hacks for long words at sentence and segment boundaries. All on the
//! host: the attention of six heads over a window is small (the q rows and the encoder's keys come down).

use crate::text::{Vocab, TOKENS_PER_SECOND, UNSPACED};

/// A word and when it is said (seconds from the window's start, until placed)
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub word: String,
    pub tokens: Vec<u32>,
    pub start: f64,
    pub end: f64,
    pub probability: f32,
}

pub const PREPEND_PUNCTUATIONS: &str = "\"'\u{201c}\u{bf}([{-";
pub const APPEND_PUNCTUATIONS: &str = "\"'.\u{3002},\u{ff0c}!\u{ff01}?\u{ff1f}:\u{ff1a}\u{201d})]}\u{3001}";
const PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// A median filter of `width` (odd) along each row of x [rows, m], reflect-padded (scipy / torch's as openai uses
/// them); rows no longer than width / 2 are left as they are
pub fn median_filter(x: &mut [f32], m: usize, width: usize) {
    let pad = width / 2;
    if m <= pad || pad == 0 {
        return;
    }
    let mut row = vec![0f32; m + 2 * pad];
    let mut win = vec![0f32; width];
    for r in x.chunks_exact_mut(m) {
        for (i, v) in row.iter_mut().enumerate() {
            let j = i as isize - pad as isize;
            let j = if j < 0 { -j } else if j >= m as isize { 2 * (m as isize - 1) - j } else { j };
            *v = r[j as usize];
        }
        for (i, o) in r.iter_mut().enumerate() {
            win.copy_from_slice(&row[i..i + width]);
            win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            *o = win[pad];
        }
    }
}

/// Dynamic time warping over a cost x [n, m] (openai's dtw_cpu and backtrace): the path's (token, frame) pairs in
/// order
pub fn dtw(x: &[f32], n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let w = m + 1;
    let mut cost = vec![f32::INFINITY; (n + 1) * w];
    let mut trace = vec![-1i8; (n + 1) * w];
    cost[0] = 0.0;
    for j in 1..=m {
        for i in 1..=n {
            let (c0, c1, c2) = (cost[(i - 1) * w + j - 1], cost[(i - 1) * w + j], cost[i * w + j - 1]);
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            cost[i * w + j] = x[(i - 1) * m + j - 1] + c;
            trace[i * w + j] = t;
        }
    }
    trace[..w].fill(2);
    for i in 0..=n {
        trace[i * w] = 1;
    }
    let (mut i, mut j) = (n, m);
    let mut path = Vec::new();
    while i > 0 || j > 0 {
        path.push((i.wrapping_sub(1), j.wrapping_sub(1)));
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path.into_iter().filter(|(a, b)| *a != usize::MAX && *b != usize::MAX).unzip()
}

/// tokenizer.split_tokens_on_unicode: tokens grouped until they decode to whole characters
pub fn split_on_unicode(tokens: &[u32], v: &Vocab) -> (Vec<String>, Vec<Vec<u32>>) {
    let full: Vec<char> = v.tok.decode(tokens).chars().collect();
    let (mut words, mut groups) = (Vec::new(), Vec::new());
    let mut cur: Vec<u32> = Vec::new();
    let mut offset = 0usize;
    for t in tokens {
        cur.push(*t);
        let d = v.tok.decode(&cur);
        let chars: Vec<char> = d.chars().collect();
        let whole = match chars.iter().position(|c| *c == '\u{fffd}') {
            None => true,
            Some(p) => full.get(offset + p) == Some(&'\u{fffd}'),
        };
        if whole {
            offset += chars.len();
            words.push(d);
            groups.push(std::mem::take(&mut cur));
        }
    }
    (words, groups)
}

/// tokenizer.split_to_word_tokens: at spaces and punctuation, or (languages without spaces) at characters
pub fn split_words(tokens: &[u32], v: &Vocab, language: &str) -> (Vec<String>, Vec<Vec<u32>>) {
    let (subs, groups) = split_on_unicode(tokens, v);
    if UNSPACED.contains(&language) {
        return (subs, groups);
    }
    let (mut words, mut wt): (Vec<String>, Vec<Vec<u32>>) = (Vec::new(), Vec::new());
    for (s, g) in subs.into_iter().zip(groups) {
        let special = g[0] >= v.eot;
        let with_space = s.starts_with(' ');
        // (Python's `in`: a substring of the punctuation, the empty string too)
        let punctuation = PUNCTUATION.contains(s.trim());
        if special || with_space || punctuation || words.is_empty() {
            words.push(s);
            wt.push(g);
        } else {
            words.last_mut().unwrap().push_str(&s);
            wt.last_mut().unwrap().extend(g);
        }
    }
    (words, wt)
}

/// merge_punctuations: opening marks onto the word after, closing ones onto the word before
pub fn merge_punctuations(a: &mut [Word]) {
    if a.len() < 2 {
        return;
    }
    let (mut i, mut j) = (a.len() as isize - 2, a.len() - 1);
    while i >= 0 {
        let iu = i as usize;
        if a[iu].word.starts_with(' ') && !a[iu].word.trim().is_empty() && PREPEND_PUNCTUATIONS.contains(a[iu].word.trim()) {
            let (w, t) = (std::mem::take(&mut a[iu].word), std::mem::take(&mut a[iu].tokens));
            a[j].word = w + &a[j].word;
            let mut nt = t;
            nt.extend(a[j].tokens.iter().copied());
            a[j].tokens = nt;
        } else {
            j = iu;
        }
        i -= 1;
    }
    let (mut i, mut j) = (0usize, 1usize);
    while j < a.len() {
        if !a[i].word.ends_with(' ') && !a[j].word.is_empty() && APPEND_PUNCTUATIONS.contains(a[j].word.as_str()) {
            let (w, t) = (std::mem::take(&mut a[j].word), std::mem::take(&mut a[j].tokens));
            a[i].word.push_str(&w);
            a[i].tokens.extend(t);
        } else {
            i = j;
        }
        j += 1;
    }
}

/// find_alignment's arithmetic: `qk` holds each alignment head's raw scaled attention logits [rows, m_full] (the
/// tokens fed: the start of transcript sequence, <|notimestamps|>, the text, <|endoftext|>), `num_frames` the
/// window's frames of sound, `probs` the text tokens' probabilities. The words, timed from the window's start.
#[allow(clippy::too_many_arguments)]
pub fn find_alignment(qk: &[Vec<f32>], rows: usize, m_full: usize, num_frames: usize, sot_len: usize, text: &[u32], probs: &[f32], v: &Vocab,
                      language: &str, medfilt: usize) -> Vec<Word> {
    let m = (num_frames / 2).min(m_full).max(1);
    if qk.is_empty() || text.is_empty() {
        return Vec::new();
    }
    let mut mean = vec![0f32; rows * m];
    for h in qk {
        let mut w = vec![0f32; rows * m];
        for r in 0..rows {
            let src = &h[r * m_full..r * m_full + m];
            let mx = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = src.iter().map(|x| ((x - mx) as f64).exp()).sum();
            for (o, x) in w[r * m..(r + 1) * m].iter_mut().zip(src) {
                *o = (((x - mx) as f64).exp() / z) as f32;
            }
        }
        // normalized across the tokens, a frame at a time
        for c in 0..m {
            let (mut s, mut s2) = (0f64, 0f64);
            for r in 0..rows {
                let x = w[r * m + c] as f64;
                s += x;
                s2 += x * x;
            }
            let mu = s / rows as f64;
            let sd = (s2 / rows as f64 - mu * mu).max(0.0).sqrt().max(1e-12);
            for r in 0..rows {
                w[r * m + c] = ((w[r * m + c] as f64 - mu) / sd) as f32;
            }
        }
        median_filter(&mut w, m, medfilt);
        for (a, b) in mean.iter_mut().zip(&w) {
            *a += b / qk.len() as f32;
        }
    }
    // the rows from <|notimestamps|> to the last text token
    let r0 = sot_len;
    let r1 = rows - 1;
    let n = r1 - r0;
    let cost: Vec<f32> = mean[r0 * m..r1 * m].iter().map(|x| -x).collect();
    let (ti, fi) = dtw(&cost, n, m);
    let mut toks = text.to_vec();
    toks.push(v.eot);
    let (words, wt) = split_words(&toks, v, language);
    if wt.len() <= 1 {
        return Vec::new();
    }
    let mut bounds = vec![0usize];
    for g in &wt[..wt.len() - 1] {
        bounds.push(bounds.last().unwrap() + g.len());
    }
    let jump_times: Vec<f64> = (0..ti.len()).filter(|&k| k == 0 || ti[k] != ti[k - 1]).map(|k| fi[k] as f64 / TOKENS_PER_SECOND).collect();
    let mut out = Vec::new();
    for k in 0..words.len().min(bounds.len() - 1) {
        let (b0, b1) = (bounds[k], bounds[k + 1]);
        let (Some(s), Some(e)) = (jump_times.get(b0), jump_times.get(b1)) else { break };
        let p = &probs[b0.min(probs.len())..b1.min(probs.len())];
        let prob = if p.is_empty() { 0.0 } else { p.iter().sum::<f32>() / p.len() as f32 };
        out.push(Word { word: words[k].clone(), tokens: wt[k].clone(), start: *s, end: *e, probability: prob });
    }
    out
}

/// A segment as it is given back (absolute seconds)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Placed {
    pub start: f64,
    pub end: f64,
    pub text_tokens: Vec<u32>,
    pub words: Vec<Word>,
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// add_word_timestamps after find_alignment: the long-word hacks, the punctuation merged, the words handed to their
/// segments (by token counts) at `offset` seconds, the segments' ends moved to their words'. `last_speech`: the end
/// of the speech before (carried window to window); its new value is returned.
pub fn place_words(segments: &mut [Placed], mut a: Vec<Word>, offset: f64, mut last_speech: f64) -> f64 {
    let mut durations: Vec<f64> = a.iter().map(|w| w.end - w.start).filter(|d| *d != 0.0).collect();
    let median = if durations.is_empty() {
        0.0
    } else {
        durations.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let n = durations.len();
        if n % 2 == 1 { durations[n / 2] } else { (durations[n / 2 - 1] + durations[n / 2]) / 2.0 }
    };
    let median = median.min(0.7);
    let max_d = median * 2.0;
    if !durations.is_empty() {
        let ends = ".\u{3002}!\u{ff01}?\u{ff1f}";
        for i in 1..a.len() {
            if a[i].end - a[i].start > max_d {
                if !a[i].word.is_empty() && ends.contains(a[i].word.as_str()) {
                    a[i].end = a[i].start + max_d;
                } else if !a[i - 1].word.is_empty() && ends.contains(a[i - 1].word.as_str()) {
                    a[i].start = a[i].end - max_d;
                }
            }
        }
    }
    merge_punctuations(&mut a);
    let mut wi = 0;
    for seg in segments.iter_mut() {
        let mut saved = 0;
        let mut words: Vec<Word> = Vec::new();
        while wi < a.len() && saved < seg.text_tokens.len() {
            let t = &a[wi];
            if !t.word.is_empty() {
                words.push(Word { start: round2(offset + t.start), end: round2(offset + t.end), ..t.clone() });
            }
            saved += t.tokens.len();
            wi += 1;
        }
        if !words.is_empty() {
            if words[0].end - last_speech > median * 4.0
                && (words[0].end - words[0].start > max_d || (words.len() > 1 && words[1].end - words[0].start > max_d * 2.0))
            {
                if words.len() > 1 && words[1].end - words[1].start > max_d {
                    let boundary = (words[1].end / 2.0).max(words[1].end - max_d);
                    words[0].end = boundary;
                    words[1].start = boundary;
                }
                words[0].start = (words[0].end - max_d).max(0.0);
            }
            if seg.start < words[0].end && seg.start - 0.5 > words[0].start {
                words[0].start = (words[0].end - median).min(seg.start).max(0.0);
            } else {
                seg.start = words[0].start;
            }
            let last = words.len() - 1;
            if seg.end > words[last].start && seg.end + 0.5 < words[last].end {
                words[last].end = (words[last].start + median).max(seg.end);
            } else {
                seg.end = words[last].end;
            }
            last_speech = seg.end;
        }
        seg.words = words;
    }
    last_speech
}

#[cfg(test)]
mod tests {
    use super::*;

    /// find_alignment on openai's own attention of a window (NS_TEST_WHISPER_REF: reference/whisper/ref.py's
    /// align_qk.npy, align_tokens.npy, align.json) gives openai's words and times
    #[test]
    fn the_alignment_is_openais() {
        let (Some(v), Ok(dir)) = (crate::text::tests::vocab(), std::env::var("NS_TEST_WHISPER_REF")) else { return };
        let mut checked = 0;
        for clip in ["short", "long", "song"] {
            let d = std::path::Path::new(&dir).join(clip);
            let Ok((shape, qk)) = crate::check::read_npy(&d.join("align_qk.npy")) else { continue };
            let (h, rows, m_full) = (shape[0], shape[1], shape[2]);
            let qk: Vec<Vec<f32>> = qk.chunks(rows * m_full).map(|c| c.iter().map(|x| *x as f32).collect()).collect();
            assert_eq!(qk.len(), h);
            let fed: Vec<u32> = crate::check::read_npy(&d.join("align_tokens.npy")).unwrap().1.iter().map(|x| *x as u32).collect();
            let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("align.json")).unwrap()).unwrap();
            let sot = r["sot_len"].as_u64().unwrap() as usize;
            let text = &fed[sot + 1..fed.len() - 1];
            let probs = vec![0f32; text.len()];
            let got = find_alignment(&qk, rows, m_full, r["num_frames"].as_u64().unwrap() as usize, sot, text, &probs, &v,
                                     r["language"].as_str().unwrap(), 7);
            let want = r["words"].as_array().unwrap();
            assert_eq!(got.len(), want.len(), "{clip}: words");
            for (g, w) in got.iter().zip(want) {
                assert_eq!(g.word, w["word"].as_str().unwrap(), "{clip}");
                let wt: Vec<u32> = w["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
                assert_eq!(g.tokens, wt, "{clip}: {}", g.word);
                assert!((g.start - w["start"].as_f64().unwrap()).abs() < 1e-6 && (g.end - w["end"].as_f64().unwrap()).abs() < 1e-6,
                        "{clip}: {:?} {}..{}, openai {}..{}", g.word, g.start, g.end, w["start"], w["end"]);
            }
            checked += 1;
        }
        assert_eq!(checked, 3, "the three clips");
    }

    /// A one-window recording end to end on the host: openai's tokens cut into segments, its alignment's words placed,
    /// give its transcribe's segments and words (NS_TEST_WHISPER_REF's short clip)
    #[test]
    fn a_window_placed_is_openais_transcript() {
        let (Some(v), Ok(dir)) = (crate::text::tests::vocab(), std::env::var("NS_TEST_WHISPER_REF")) else { return };
        let d = std::path::Path::new(&dir).join("short");
        let toks: Vec<u32> = crate::check::read_npy(&d.join("tokens.npy")).unwrap().1.iter().map(|x| *x as u32).collect();
        let a: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("align.json")).unwrap()).unwrap();
        let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("result.json")).unwrap()).unwrap();
        let frames = a["num_frames"].as_u64().unwrap() as usize;
        let (cuts, _) = crate::text::cut(&toks[3..], &v, frames);
        let mut placed: Vec<Placed> = cuts.into_iter().map(|c| {
            let text: Vec<u32> = c.tokens.iter().copied().filter(|t| *t < v.eot).collect();
            let empty = c.start == c.end || v.text(&text).trim().is_empty();
            Placed { start: c.start, end: c.end, text_tokens: if empty { Vec::new() } else { text }, words: Vec::new() }
        }).collect();
        let words: Vec<Word> = a["words"].as_array().unwrap().iter().map(|w| Word {
            word: w["word"].as_str().unwrap().into(), tokens: w["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(),
            start: w["start"].as_f64().unwrap(), end: w["end"].as_f64().unwrap(), probability: w["probability"].as_f64().unwrap() as f32 }).collect();
        place_words(&mut placed, words, 0.0, 0.0);
        let placed: Vec<&Placed> = placed.iter().filter(|p| !p.text_tokens.is_empty()).collect();
        let want = r["segments"].as_array().unwrap();
        assert_eq!(placed.len(), want.len(), "segments");
        for (g, w) in placed.iter().zip(want) {
            assert_eq!(v.text(&g.text_tokens), w["text"].as_str().unwrap());
            assert!((g.start - w["start"].as_f64().unwrap()).abs() < 1e-6 && (g.end - w["end"].as_f64().unwrap()).abs() < 1e-6,
                    "segment {}..{}, openai {}..{}", g.start, g.end, w["start"], w["end"]);
            let ww = w["words"].as_array().unwrap();
            assert_eq!(g.words.len(), ww.len(), "{:?}", w["text"]);
            for (x, y) in g.words.iter().zip(ww) {
                assert_eq!(x.word, y["word"].as_str().unwrap());
                assert!((x.start - y["start"].as_f64().unwrap()).abs() < 1e-6 && (x.end - y["end"].as_f64().unwrap()).abs() < 1e-6,
                        "{:?} {}..{}, openai {}..{}", x.word, x.start, x.end, y["start"], y["end"]);
            }
        }
    }

    #[test]
    fn dtw_follows_the_cheap_diagonal() {
        // 3 tokens over 6 frames: each token cheap on two frames in turn
        let n = 3;
        let m = 6;
        let mut x = vec![1f32; n * m];
        for (i, f) in [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)] {
            x[i * m + f] = 0.0;
        }
        let (ti, fi) = dtw(&x, n, m);
        assert_eq!(ti, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(fi, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn dtw_is_openais_on_a_mixed_cost() {
        // numpy: whisper.timing.dtw_cpu(-np.array([[3,1,0,0],[0,2,2,0],[0,0,1,3]], dtype=np.float32))
        let x: Vec<f32> = [3., 1., 0., 0., 0., 2., 2., 0., 0., 0., 1., 3.].iter().map(|v: &f32| -v).collect();
        let (ti, fi) = dtw(&x, 3, 4);
        assert_eq!(ti, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(fi, vec![0, 1, 1, 2, 2, 3]);
    }

    #[test]
    fn the_median_filter_reflects_at_the_edges() {
        let mut x = vec![5., 1., 9., 2., 8., 3., 7.];
        median_filter(&mut x, 7, 3);
        // reflect: [1, 5,1,9,2,8,3,7, 3]
        assert_eq!(x, vec![1., 5., 2., 8., 3., 7., 3.]);
        let mut short = vec![1., 2.];
        median_filter(&mut short, 2, 7);
        assert_eq!(short, vec![1., 2.], "a row shorter than the pad is kept");
    }

    fn w(word: &str, start: f64, end: f64) -> Word {
        Word { word: word.into(), tokens: vec![1], start, end, probability: 1.0 }
    }

    #[test]
    fn punctuation_joins_its_word() {
        let mut a = vec![w(" \"", 0.0, 0.1), w(" Hello", 0.1, 0.5), w(",", 0.5, 0.6), w(" world", 0.6, 1.0), w("!", 1.0, 1.1), w("\"", 1.1, 1.2)];
        merge_punctuations(&mut a);
        let words: Vec<&str> = a.iter().map(|x| x.word.as_str()).filter(|x| !x.is_empty()).collect();
        assert_eq!(words, vec![" \" Hello,", " world!\""]);
        assert_eq!(a[1].tokens.len(), 3);
    }

    #[test]
    fn words_go_to_their_segments_and_the_segments_follow_them() {
        let mut segs = vec![Placed { start: 10.0, end: 12.0, text_tokens: vec![1, 1], words: vec![] },
                            Placed { start: 12.0, end: 14.0, text_tokens: vec![1], words: vec![] }];
        let a = vec![w(" one", 0.2, 0.6), w(" two", 0.6, 1.0), w(" three", 2.1, 2.5)];
        let last = place_words(&mut segs, a, 10.0, 10.0);
        assert_eq!(segs[0].words.iter().map(|x| x.word.as_str()).collect::<Vec<_>>(), vec![" one", " two"]);
        assert_eq!((segs[0].start, segs[0].end), (10.2, 11.0));
        assert_eq!((segs[1].start, segs[1].end), (12.1, 12.5));
        assert_eq!(last, 12.5);
    }

    #[test]
    fn a_long_word_after_a_sentence_end_is_cut() {
        let mut segs = vec![Placed { start: 0.0, end: 9.0, text_tokens: vec![1, 1, 1, 1], words: vec![] }];
        let a = vec![w(" Hi", 0.0, 0.4), w(".", 0.4, 0.4), w(" So", 0.4, 5.0), w(" yes", 5.0, 5.4)];
        place_words(&mut segs, a, 0.0, 0.0);
        let so = segs[0].words.iter().find(|x| x.word == " So").unwrap();
        // the median of the non-zero durations 0.4, 4.6, 0.4: 0.4; at most twice it
        assert!((so.start - 4.2).abs() < 1e-9 && so.end == 5.0, "{so:?}");
    }
}
