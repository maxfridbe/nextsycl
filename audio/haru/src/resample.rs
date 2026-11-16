//! Resampling to and from the model's 24 kHz: a windowed sinc (Hann, 16 zero crossings, cut at the lower Nyquist),
//! whole (a voice reference) or streaming (the microphone, as it comes: each output waits for the input its taps
//! reach, so chunks join without seams).

fn tap(d: f64, cut: f64, zc: f64) -> f64 {
    let d = d * cut;
    if d.abs() >= zc {
        return 0.0;
    }
    let sinc = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d) };
    sinc * (0.5 + 0.5 * (std::f64::consts::PI * d / zc).cos()) * cut
}

const ZC: f64 = 16.0;

/// Mono samples at `rate` to `to`, whole
pub fn resample(x: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || x.is_empty() || rate == 0 {
        return x.to_vec();
    }
    let mut s = Stream::new(rate, to);
    let mut out = s.push(x);
    out.extend(s.flush());
    out.truncate((x.len() as u64 * to as u64 / rate as u64) as usize);
    out
}

/// A streaming resampler: `push` what comes, get what is ready
pub struct Stream {
    rate: u32,
    to: u32,
    /// the ratio as l / m outputs an input (reduced), and each phase's taps (input c - half + k for output phase p)
    l: u64,
    m: u64,
    half: i64,
    table: Vec<f32>,
    /// input samples kept, the first at absolute index `base`
    buf: Vec<f32>,
    base: i64,
    /// the next output's index
    next: u64,
}

impl Stream {
    pub fn new(rate: u32, to: u32) -> Stream {
        let cut = (to as f64 / rate.max(1) as f64).min(1.0);
        let (mut a, mut b) = (rate.max(1) as u64, to.max(1) as u64);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let (l, m) = (to.max(1) as u64 / a, rate.max(1) as u64 / a);
        let half = (ZC / cut).ceil() as i64;
        let width = (2 * half + 2) as usize;
        // a table a phase while the ratio is a small fraction (every common rate), else none (each tap computed)
        let mut table = Vec::new();
        if l <= 4096 {
            table = vec![0f32; l as usize * width];
            for p in 0..l as usize {
                for k in 0..width {
                    table[p * width + k] = tap((k as i64 - half) as f64 - p as f64 / l as f64, cut, ZC) as f32;
                }
            }
        }
        Stream { rate, to, l, m, half, table, buf: Vec::new(), base: 0, next: 0 }
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    fn produce(&mut self, end: Option<i64>) -> Vec<f32> {
        if self.rate == self.to {
            return std::mem::take(&mut self.buf);
        }
        let have = self.base + self.buf.len() as i64;
        let mut out = Vec::new();
        let width = (2 * self.half + 2) as usize;
        let cut = (self.to as f64 / self.rate as f64).min(1.0);
        loop {
            // the output's instant in input samples: c + p / l
            let num = self.next * self.m;
            let (c, p) = ((num / self.l) as i64, (num % self.l) as usize);
            let reach = c + self.half + 1;
            if reach >= have && end.is_none_or(|e| c >= e) {
                break;
            }
            if let Some(e) = end {
                if c >= e {
                    break;
                }
            }
            let mut acc = 0f32;
            for k in 0..width {
                let at = c - self.half + k as i64;
                if at < self.base || at >= have {
                    continue;
                }
                let w = match self.table.is_empty() {
                    false => self.table[p * width + k],
                    true => tap((k as i64 - self.half) as f64 - p as f64 / self.l as f64, cut, ZC) as f32,
                };
                acc += self.buf[(at - self.base) as usize] * w;
            }
            out.push(acc);
            self.next += 1;
        }
        // keep what the next outputs reach back to
        let c = (self.next * self.m / self.l) as i64;
        let keep_from = (c - self.half - 1).max(self.base);
        let drop = (keep_from - self.base) as usize;
        if drop > 0 {
            self.buf.drain(..drop.min(self.buf.len()));
            self.base += drop as i64;
        }
        out
    }

    pub fn push(&mut self, x: &[f32]) -> Vec<f32> {
        self.buf.extend_from_slice(x);
        self.produce(None)
    }

    /// The outputs still owed for the input so far (its end taken as silence)
    pub fn flush(&mut self) -> Vec<f32> {
        let end = self.base + self.buf.len() as i64;
        self.produce(Some(end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_is_the_whole_and_a_tone_survives() {
        let x: Vec<f32> = (0..48000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin()).collect();
        let whole = resample(&x, 48000, 24000);
        assert_eq!(whole.len(), 24000);
        let mut s = Stream::new(48000, 24000);
        let mut parts = Vec::new();
        for c in x.chunks(1234) {
            parts.extend(s.push(c));
        }
        parts.extend(s.flush());
        parts.truncate(24000);
        assert_eq!(parts.len(), 24000);
        let d = whole.iter().zip(&parts).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(d < 1e-6, "{d}");
        // the tone at 24 kHz, away from the edges
        let e = (1000..23000).map(|i| (whole[i] - (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin()).abs()).fold(0f32, f32::max);
        assert!(e < 0.01, "{e}");
        assert_eq!(resample(&x[..16000], 16000, 24000).len(), 24000);
    }
}
