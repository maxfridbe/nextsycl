//! The sound around the model, as upstream does it (omnivoice/utils/audio.py): torchaudio's resampler, pydub's silence
//! detection (on 16-bit samples, 10 ms steps, -50 dBFS) for trimming a recording and the answer, the fade and padding at
//! the ends, the cross-fade between a long text's chunks.

/// Mono samples from `rate` to `to`: torchaudio.functional.resample (sinc, Hann window, 6 zero crossings, roll-off 0.99)
pub fn resample(x: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || x.is_empty() || rate == 0 || to == 0 {
        return x.to_vec();
    }
    let g = gcd(rate as u64, to as u64);
    let (orig, new) = ((rate as u64 / g) as usize, (to as u64 / g) as usize);
    let lw = 6.0f64;
    let base = orig.min(new) as f64 * 0.99;
    let width = (lw * orig as f64 / base).ceil() as usize;
    let k = 2 * width + orig;
    // kernel[phase][m]: t = (-phase / new + (m - width) / orig) * base, clamped to +-lw
    let mut kern = vec![0f32; new * k];
    for p in 0..new {
        for m in 0..k {
            let idx = (m as f64 - width as f64) / orig as f64;
            let t = ((-(p as f64) / new as f64 + idx) * base).clamp(-lw, lw);
            let window = (t * std::f64::consts::PI / lw / 2.0).cos().powi(2);
            let tp = t * std::f64::consts::PI;
            let s = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
            kern[p * k + m] = (s * window * base / orig as f64) as f32;
        }
    }
    let n = (new as u64 * x.len() as u64).div_ceil(orig as u64) as usize;
    let at = |i: isize| -> f32 { if i < 0 || i as usize >= x.len() { 0.0 } else { x[i as usize] } };
    (0..n).map(|i| {
        let (j, p) = (i / new, i % new);
        let s0 = (j * orig) as isize - width as isize;
        let kr = &kern[p * k..(p + 1) * k];
        kr.iter().enumerate().map(|(m, w)| w * at(s0 + m as isize)).sum()
    }).collect()
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// pydub's view of a sound: 16-bit samples (numpy's astype: truncated toward zero), its length in whole milliseconds
pub struct Seg {
    pub s: Vec<i16>,
    rate: usize,
}

impl Seg {
    pub fn of(x: &[f32], rate: u32) -> Seg {
        Seg { s: x.iter().map(|v| (v * 32768.0).clamp(-32768.0, 32767.0) as i16).collect(), rate: rate as usize }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        self.s.iter().map(|v| *v as f32 / 32768.0).collect()
    }

    /// len(segment): milliseconds, rounded
    fn ms(&self) -> usize {
        (self.s.len() as f64 * 1000.0 / self.rate as f64).round() as usize
    }

    /// the frame of a millisecond (int(ms * rate / 1000))
    fn frame(&self, ms: usize) -> usize {
        ((ms as f64 * self.rate as f64 / 1000.0) as usize).min(self.s.len())
    }

    /// segment[a:b] in milliseconds (clamped to the length)
    fn slice(&self, a: usize, b: usize) -> &[i16] {
        let (a, b) = (self.frame(a.min(self.ms())), self.frame(b.min(self.ms())));
        &self.s[a..b.max(a)]
    }

    fn sub(&self, a: usize, b: usize) -> Seg {
        Seg { s: self.slice(a, b).to_vec(), rate: self.rate }
    }
}

/// audioop.rms: the root mean square, truncated to an integer
fn rms(s: &[i16]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let sum: f64 = s.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    (sum / s.len() as f64).sqrt().floor()
}

/// dBFS of samples (-inf for silence)
fn dbfs(s: &[i16]) -> f64 {
    let r = rms(s);
    if r == 0.0 { f64::NEG_INFINITY } else { 20.0 * (r / 32768.0).log10() }
}

/// pydub.silence.detect_silence: [start, end] ms of stretches at least `min_len` long at or below `thresh` dBFS
fn detect_silence(seg: &Seg, min_len: usize, thresh: f64, step: usize) -> Vec<[usize; 2]> {
    let len = seg.ms();
    if len < min_len {
        return Vec::new();
    }
    let t = 10f64.powf(thresh / 20.0) * 32768.0;
    let last = len - min_len;
    let mut starts: Vec<usize> = (0..=last).step_by(step).collect();
    if !last.is_multiple_of(step) {
        starts.push(last);
    }
    let silent: Vec<usize> = starts.into_iter().filter(|i| rms(seg.slice(*i, i + min_len)) <= t).collect();
    let Some((&first, rest)) = silent.split_first() else { return Vec::new() };
    let mut out = Vec::new();
    let (mut prev, mut start) = (first, first);
    for &i in rest {
        let continuous = i == prev + step;
        let gap = i > prev + min_len;
        if !continuous && gap {
            out.push([start, prev + min_len]);
            start = i;
        }
        prev = i;
    }
    out.push([start, prev + min_len]);
    out
}

/// pydub.silence.detect_nonsilent
fn detect_nonsilent(seg: &Seg, min_len: usize, thresh: f64, step: usize) -> Vec<[usize; 2]> {
    let silent = detect_silence(seg, min_len, thresh, step);
    let len = seg.ms();
    if silent.is_empty() {
        return vec![[0, len]];
    }
    if silent[0] == [0, len] {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut prev_end = 0;
    let mut end = 0;
    for [s, e] in &silent {
        out.push([prev_end, *s]);
        prev_end = *e;
        end = *e;
    }
    if end != len {
        out.push([prev_end, len]);
    }
    if out[0] == [0, 0] {
        out.remove(0);
    }
    out
}

/// pydub.silence.split_on_silence, the pieces joined again: silences over `min_len` ms shortened to `keep` ms a side
fn split_join(seg: &Seg, min_len: usize, thresh: f64, keep: usize, step: usize) -> Seg {
    let mut ranges: Vec<[isize; 2]> = detect_nonsilent(seg, min_len, thresh, step).iter().map(|[a, b]| [*a as isize - keep as isize, (*b + keep) as isize]).collect();
    for i in 1..ranges.len() {
        let (last_end, next_start) = (ranges[i - 1][1], ranges[i][0]);
        if next_start < last_end {
            let mid = (last_end + next_start).div_euclid(2);
            ranges[i - 1][1] = mid;
            ranges[i][0] = mid;
        }
    }
    let len = seg.ms() as isize;
    let mut s = Vec::new();
    for [a, b] in ranges {
        s.extend_from_slice(seg.slice(a.max(0) as usize, b.min(len).max(0) as usize));
    }
    Seg { s, rate: seg.rate }
}

/// pydub.silence.detect_leading_silence: ms of 10 ms chunks below `thresh` dBFS at the start
fn leading_silence(seg: &Seg, thresh: f64) -> usize {
    let len = seg.ms();
    let mut t = 0;
    while t < len && dbfs(seg.slice(t, t + 10)) < thresh {
        t += 10;
    }
    t.min(len)
}

/// upstream's remove_silence: middle silences over `mid` ms (0: kept) shortened to `mid` ms a side, the ends trimmed to
/// `lead` / `trail` ms of silence (at -50 dBFS, 10 ms steps); the samples come back through 16 bits, as upstream's
pub fn remove_silence(x: &[f32], rate: u32, mid: usize, lead: usize, trail: usize) -> Vec<f32> {
    let mut seg = Seg::of(x, rate);
    if mid > 0 {
        seg = split_join(&seg, mid, -50.0, mid, 10);
    }
    let start = leading_silence(&seg, -50.0).saturating_sub(lead);
    seg = seg.sub(start, seg.ms());
    seg.s.reverse();
    let start = leading_silence(&seg, -50.0).saturating_sub(trail);
    seg = seg.sub(start, seg.ms());
    seg.s.reverse();
    seg.to_f32()
}

/// upstream's trim_long_audio: a recording over `threshold` s cut at the last non-silent start within `max` s (at -40
/// dBFS, 100 ms silences), not before `min` s
pub fn trim_long(x: &[f32], rate: u32, max: f32, min: f32, threshold: f32) -> Vec<f32> {
    if (x.len() as f32 / rate as f32) <= threshold {
        return x.to_vec();
    }
    let seg = Seg::of(x, rate);
    let ns = detect_nonsilent(&seg, 100, -40.0, 10);
    if ns.is_empty() {
        return x.to_vec();
    }
    let (max_ms, min_ms) = ((max * 1000.0) as usize, (min * 1000.0) as usize);
    let mut best = 0;
    for [s, e] in &ns {
        if *s > best && *s <= max_ms {
            best = *s;
        }
        if *e > max_ms {
            break;
        }
    }
    if best < min_ms {
        best = max_ms.min(seg.ms());
    }
    seg.sub(0, best).to_f32()
}

/// upstream's fade_and_pad_audio: linear fades of `fade` s at both ends (at most half the sound each), `pad` s of
/// silence each side
pub fn fade_and_pad(x: &[f32], rate: u32, pad: f32, fade: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mut y = x.to_vec();
    let k = ((fade * rate as f32) as usize).min(y.len() / 2);
    if k > 0 {
        let lin = linspace(k);
        for i in 0..k {
            y[i] *= lin[i];
            let j = y.len() - k + i;
            y[j] *= lin[k - 1 - i];
        }
    }
    let p = (pad * rate as f32) as usize;
    let mut out = vec![0f32; p];
    out.extend(y);
    out.extend(std::iter::repeat_n(0f32, p));
    out
}

/// numpy.linspace(0, 1, n) in float32
fn linspace(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![0.0];
    }
    (0..n).map(|i| (i as f64 / (n - 1) as f64) as f32).collect()
}

/// upstream's cross_fade_chunks: each chunk faded out over 0.1 s, 0.1 s of silence, the next faded in over 0.1 s
pub fn cross_fade(chunks: &[Vec<f32>], rate: u32) -> Vec<f32> {
    let Some(first) = chunks.first() else { return Vec::new() };
    let n = (0.3 * rate as f32) as usize / 3;
    let mut merged = first.clone();
    for c in &chunks[1..] {
        let fo = n.min(merged.len());
        if fo > 0 {
            let lin = linspace(fo);
            let l = merged.len();
            for i in 0..fo {
                merged[l - fo + i] *= lin[fo - 1 - i];
            }
        }
        merged.extend(std::iter::repeat_n(0f32, n));
        let mut c = c.clone();
        let fi = n.min(c.len());
        if fi > 0 {
            let lin = linspace(fi);
            for i in 0..fi {
                c[i] *= lin[i];
            }
        }
        merged.extend(c);
    }
    merged
}

/// The answer's level: a clone's like its recording's (scaled by its RMS / 0.1 when that was under 0.1), otherwise
/// peak-normalized to 0.5
pub fn level(x: &mut [f32], ref_rms: Option<f32>) {
    match ref_rms {
        Some(r) if r < 0.1 => x.iter_mut().for_each(|v| *v *= r / 0.1),
        Some(_) => {}
        None => {
            let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            if peak > 1e-6 {
                x.iter_mut().for_each(|v| *v = *v / peak * 0.5);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_resampler_keeps_a_tone_and_the_length() {
        let x: Vec<f32> = (0..24000).map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 24000.0).sin()).collect();
        let y = resample(&x, 24000, 16000);
        assert_eq!(y.len(), 16000);
        // the same tone at 16 kHz, away from the edges
        for (i, got) in y.iter().enumerate().take(1100).skip(1000) {
            let want = (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16000.0).sin();
            assert!((got - want).abs() < 0.01, "{i}: {got} vs {want}");
        }
        assert_eq!(resample(&x, 24000, 24000), x);
        assert_eq!(resample(&x[..16000], 16000, 24000).len(), 24000);
    }

    #[test]
    fn silence_is_cut_to_what_upstream_keeps() {
        // 0.5 s of tone, 1.5 s of silence, 0.5 s of tone, at 24 kHz
        let rate = 24000;
        let tone = |n: usize| (0..n).map(|i| 0.3 * (i as f32 * 0.1).sin()).collect::<Vec<f32>>();
        let mut x = tone(12000);
        x.extend(vec![0.0; 36000]);
        x.extend(tone(12000));
        let y = remove_silence(&x, rate, 500, 100, 100);
        // the middle 1.5 s comes down to about 0.5 s a side, at most the gap
        let secs = y.len() as f32 / rate as f32;
        assert!(secs > 1.9 && secs < 2.6, "{secs}");
        // edges: leading and trailing silence trimmed to 100 ms
        let mut z = vec![0.0; 24000];
        z.extend(tone(12000));
        z.extend(vec![0.0; 24000]);
        let w = remove_silence(&z, rate, 0, 100, 100);
        let secs = w.len() as f32 / rate as f32;
        assert!((secs - 0.7).abs() < 0.03, "{secs}");
    }

    #[test]
    fn fades_and_pads_as_upstream() {
        let y = fade_and_pad(&[1.0; 10], 100, 0.02, 0.03);
        assert_eq!(y.len(), 14);
        assert_eq!(&y[..2], &[0.0, 0.0]);
        assert_eq!(y[2], 0.0);
        assert_eq!(y[3], 0.5);
        assert_eq!(y[4], 1.0);
        assert_eq!(y[11], 0.0);
    }
}
