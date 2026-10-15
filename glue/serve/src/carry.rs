//! Speech that carries on from speech made before (a book read in chunks): the end of the earlier chunk - its last
//! sentences, cut at the pause before them - is the in-context reference the next chunk is cloned from, after the
//! voice's own sample when there is one, so the voice and its pace go on and the chain is anchored to the voice.

use nextsycl_audio::Reference;

/// The most of the earlier chunk kept (its last sentences within this, at least the last one)
pub const KEEP_S: f64 = 15.0;

/// The end of `samples` (mono, `rate`) that says the last sentences of `text`, within `keep_s` seconds (at least the
/// last sentence): (its samples, its text). The cut is placed by the characters before it and moved to the quietest
/// moment near there (the pause between sentences).
pub fn tail(samples: &[f32], rate: u32, text: &str, keep_s: f64) -> (Vec<f32>, String) {
    let dur = samples.len() as f64 / rate.max(1) as f64;
    let text = text.trim();
    if dur <= keep_s {
        return (samples.to_vec(), text.to_string());
    }
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len().max(1) as f64;
    // where each sentence after the first begins (a character index)
    let starts: Vec<usize> = (1..chars.len())
        .filter(|&i| chars[i - 1].is_whitespace() && ends_sentence(&chars[..i - 1]) && !chars[i].is_whitespace())
        .collect();
    // the earliest start whose remainder fits, else the last one
    let Some(&at) = starts.iter().find(|&&i| dur * (1.0 - i as f64 / total) <= keep_s).or(starts.last()) else {
        return (samples.to_vec(), text.to_string());
    };
    let guess = dur * at as f64 / total;
    let cut = quietest(samples, rate, guess, 0.7);
    (samples[cut..].to_vec(), chars[at..].iter().collect::<String>().trim().to_string())
}

/// Whether the text before ends a sentence (closing quotes and brackets after the mark allowed)
fn ends_sentence(before: &[char]) -> bool {
    let t: Vec<char> = before.iter().rev().copied().skip_while(|c| matches!(c, '"' | '\'' | ')' | ']' | '”' | '’' | '»')).take(1).collect();
    t.first().is_some_and(|c| matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？' | ';' | ':'))
}

/// The sample where the quietest 40 ms window within `span` seconds of `at` begins
fn quietest(samples: &[f32], rate: u32, at: f64, span: f64) -> usize {
    let r = rate.max(1) as f64;
    let win = (0.04 * r) as usize;
    let hop = (win / 4).max(1);
    let lo = ((at - span).max(0.0) * r) as usize;
    let hi = (((at + span) * r) as usize).min(samples.len().saturating_sub(win));
    let mut best = ((at * r) as usize).min(samples.len().saturating_sub(1), );
    let mut quiet = f32::MAX;
    let mut i = lo;
    while i < hi {
        let e: f32 = samples[i..i + win].iter().map(|v| v * v).sum();
        if e < quiet {
            quiet = e;
            best = i + win / 2;
        }
        i += hop;
    }
    best
}

/// A reference that says `a` then `b` (b resampled to a's rate; a short pause between)
pub fn join(a: Reference, b: Reference) -> Reference {
    let mut samples = a.samples;
    samples.extend(std::iter::repeat_n(0.0, (a.rate as f64 * 0.3) as usize));
    samples.extend(resample(&b.samples, b.rate, a.rate));
    let text = match (a.text, b.text) {
        (Some(x), Some(y)) => Some(format!("{} {}", x.trim(), y.trim())),
        (x, y) => x.or(y),
    };
    Reference { samples, rate: a.rate, text }
}

/// Linear resampling (a reference's, where a little softness does not matter)
fn resample(x: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as f64 * to as f64 / rate as f64) as usize;
    (0..n).map(|i| {
        let p = i as f64 * rate as f64 / to as f64;
        let k = p as usize;
        let f = (p - k as f64) as f32;
        x[k.min(x.len() - 1)] * (1.0 - f) + x[(k + 1).min(x.len() - 1)] * f
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a tone a sentence, a silence after each: (samples, rate)
    fn spoken(seconds: &[f64]) -> (Vec<f32>, u32) {
        let rate = 1000;
        let mut s = Vec::new();
        for d in seconds {
            s.extend((0..(d * rate as f64) as usize).map(|i| (i as f32 * 0.3).sin() * 0.5));
            s.extend(std::iter::repeat_n(0.0, 300));
        }
        (s, rate)
    }

    #[test]
    fn a_short_chunk_is_kept_whole() {
        let (s, r) = spoken(&[3.0, 3.0]);
        let (t, text) = tail(&s, r, "One two. Three four.", 15.0);
        assert_eq!(t.len(), s.len());
        assert_eq!(text, "One two. Three four.");
    }

    #[test]
    fn a_long_chunk_keeps_its_last_sentences_cut_at_a_pause() {
        // three sentences of equal length, 8 s each: the last fits in 10 s
        let (s, r) = spoken(&[8.0, 8.0, 8.0]);
        let text = "Aaaa aaaa aaaa aaaa. Bbbb bbbb bbbb bbbb! \"Cccc cccc cccc cccc.\"";
        let (t, said) = tail(&s, r, text, 10.0);
        assert_eq!(said, "\"Cccc cccc cccc cccc.\"");
        let kept = t.len() as f64 / r as f64;
        assert!((8.0..9.0).contains(&kept), "{kept} s");
        // it starts in the silence, not mid-tone
        assert!(t[0].abs() < 1e-6);
    }

    #[test]
    fn a_reference_is_joined_at_one_rate() {
        let a = Reference { samples: vec![0.1; 2000], rate: 2000, text: Some("voice.".into()) };
        let b = Reference { samples: vec![0.2; 1000], rate: 1000, text: Some("before.".into()) };
        let j = join(a, b);
        assert_eq!(j.rate, 2000);
        assert_eq!(j.samples.len(), 2000 + 600 + 2000);
        assert_eq!(j.text.as_deref(), Some("voice. before."));
    }
}
