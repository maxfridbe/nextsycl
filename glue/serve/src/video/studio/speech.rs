//! How much of a clip is speech (the legacy `speech_pct.py`): the sound at 16 kHz mono in 30 ms windows, each
//! window's level in dB; the noise floor is the 20th percentile, the loud end the 95th, and a window is speech when
//! it is at least max(6 dB, 35% of the range) above the floor. Plus the spoken words of the prompt, for words per
//! second.

use std::path::Path;
use std::process::Command;

pub struct Speech {
    pub seconds: f64,
    pub speech_s: f64,
    pub speech_pct: f64,
    pub lead_silence: f64,
    pub tail_silence: f64,
    pub floor_db: f64,
    pub ceil_db: f64,
}

fn r(x: f64, d: i32) -> f64 {
    let m = 10f64.powi(d);
    super::pyround(x * m) / m
}

pub fn analyse(path: &Path) -> Option<Speech> {
    let out = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(path).args(["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let v: Vec<i16> = out.stdout.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    from_samples(&v)
}

pub fn from_samples(v: &[i16]) -> Option<Speech> {
    const N: usize = 480;
    if v.len() <= N {
        return None;
    }
    // windows starting at 0, N, ... below len - N (the last partial window, and an exact last one, dropped)
    let db: Vec<f64> = (0..v.len() - N)
        .step_by(N)
        .map(|i| {
            let s: f64 = v[i..i + N].iter().map(|x| (*x as f64) * (*x as f64)).sum();
            let rms = (s / N as f64).sqrt() / 32768.0;
            if rms <= 1e-9 { -120.0 } else { 20.0 * rms.log10() }
        })
        .collect();
    let mut sorted = db.clone();
    sorted.sort_by(f64::total_cmp);
    let pct = |p: f64| sorted[(super::pyround(p / 100.0 * (sorted.len() - 1) as f64) as i64).clamp(0, sorted.len() as i64 - 1) as usize];
    let (floor, ceil) = (pct(20.0), pct(95.0));
    let thr = floor + 6f64.max(0.35 * (ceil - floor));
    let talk: Vec<bool> = db.iter().map(|d| *d >= thr).collect();
    let n_talk = talk.iter().filter(|t| **t).count();
    let first = talk.iter().position(|t| *t).unwrap_or(talk.len());
    let last = talk.iter().rposition(|t| *t).map_or(talk.len(), |l| talk.len() - 1 - l);
    Some(Speech {
        seconds: r(db.len() as f64 * 0.03, 2),
        speech_s: r(n_talk as f64 * 0.03, 2),
        speech_pct: r(100.0 * n_talk as f64 / db.len() as f64, 1),
        lead_silence: r(first as f64 * 0.03, 2),
        tail_silence: r(last as f64 * 0.03, 2),
        floor_db: r(floor, 1),
        ceil_db: r(ceil, 1),
    })
}

/// Words in the prompt's spoken segments (`<d>[Language] ...</d>`).
pub fn spoken_words(prompt: &str) -> usize {
    let mut n = 0;
    let mut rest = prompt;
    while let Some(a) = rest.find("<d>[") {
        let after = &rest[a + 4..];
        let Some(close) = after.find(']') else { break };
        if !after[..close].chars().all(|c| c.is_ascii_alphabetic()) {
            rest = after;
            continue;
        }
        let body = &after[close + 1..];
        let Some(end) = body.find("</d>") else { break };
        n += body[..end].split_whitespace().count();
        rest = &body[end + 4..];
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talk_and_silence() {
        // 0.6 s silence, 0.6 s tone, 0.6 s silence at 16 kHz
        let mut v = vec![0i16; 9600];
        v.extend((0..9600).map(|i| ((i as f64 * 0.3).sin() * 8000.0) as i16));
        v.extend(vec![0i16; 9600]);
        let s = from_samples(&v).unwrap();
        assert!((s.speech_s - 0.6).abs() < 0.05, "{}", s.speech_s);
        assert!((s.lead_silence - 0.6).abs() < 0.05);
        assert_eq!(spoken_words("x <d>[English] First batch of the morning.</d> y <d>[French] Oui.</d>"), 6);
    }
}
