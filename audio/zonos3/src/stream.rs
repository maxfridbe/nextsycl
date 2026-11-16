//! Streaming decode: a take's sound handed over while it is generated.
//!
//! Rows arrive delayed (codebook c of frame f on row f + c); a frame is decodable once its last codebook arrived. The
//! decoder is run on the newly decodable frames behind `CONTEXT` frames of left context already decoded (re-decoded:
//! the codec here is stateless) and one frame is held back, whose samples the next frames' overlap-add, mixer and
//! resampler still reach; chunks ramp 4, 8, 15, 15 ... frames (Zyphra's prefix stream ramps 1, 2, 4, 8, 15; a re-decode
//! of the context makes one-frame chunks dear). At the end the rest is decoded to the take's end.
//!
//! The loudness leveling of a whole take (`level`: a 3 s window either side, smoothed both ways) cannot run on a
//! stream; `Leveler` is its causal form: the gain from the active speech of the last 3 s, smoothed forwards only.
//!
//! Origin: Zyphra's ZONOS3-Mini runtime (model/src/zonos3/tts/stream.py PrefixDecoder: the ramp and the delayed-row
//! bookkeeping; MIT).

use nextsycl_audio::Result;

/// left context re-decoded before a chunk: 20 s (the decoder's attention windows reach 10 s at 15 Hz, and further through
/// its stages)
pub const CONTEXT: usize = 300;
/// frames held back for the overlap-add / mixer / resampler tail
pub const HOLD: usize = 1;

pub fn need(emitted_generated: usize) -> usize {
    (4 + emitted_generated).min(15)
}

/// One take's prefix decode: `next(buffer, decodable)` decodes frames [emitted, decodable) behind their context
pub struct Prefix {
    /// frames of the buffer (prefill + generated) whose samples went out (the prefill's never go out)
    pub emitted: usize,
    pub np: usize,
}

impl Prefix {
    pub fn new(np: usize) -> Prefix {
        Prefix { emitted: np, np }
    }

    /// the frames to decode now: (first decoded, first emitted, end emitted, end decoded) - every complete frame is
    /// decoded, the last HOLD of them only as the right context of the ones before (None: wait); `last` with the
    /// take's final frame count
    pub fn due(&self, complete: usize, last: Option<usize>) -> Option<(usize, usize, usize, usize)> {
        let (to, end) = match last {
            Some(t) => (t, t),
            None => {
                let d = complete.saturating_sub(HOLD);
                if d < self.emitted + need(self.emitted - self.np) {
                    return None;
                }
                (d, complete)
            }
        };
        if to <= self.emitted {
            return None;
        }
        Some((self.emitted.saturating_sub(CONTEXT), self.emitted, to, end))
    }
}

/// The causal form of `level`: 50 ms frames, the active speech of the last 3 s (within 35 dB of the loud frames so
/// 10 s, above -80 dBFS) brought to `target` by a gain smoothed forwards (0.5 s), -12 to +50 dB
pub struct Leveler {
    hop: usize,
    target: f32,
    energy: Vec<f64>,
    gain_db: Option<f32>,
    carry: Vec<f32>,
}

impl Leveler {
    pub fn new(rate: u32, target: f32) -> Leveler {
        Leveler { hop: (rate as usize / 20).max(1), target, energy: Vec::new(), gain_db: None, carry: Vec::new() }
    }

    fn frame(&mut self, f: &mut [f32]) {
        let e = f.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / f.len().max(1) as f64;
        let db = |e: f64| 10.0 * (e + 1e-12).log10();
        self.energy.push(e);
        // the loud frames of the last 10 s (as `active_frames`, causal)
        let n = self.energy.len();
        let mut sorted: Vec<f64> = self.energy[n.saturating_sub(200)..].to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let loud = db(sorted[(sorted.len() / 20).min(sorted.len() - 1)]);
        let lo = n.saturating_sub(60);
        let act: Vec<f64> = self.energy[lo..].iter().copied().filter(|e| db(*e) > -80.0 && db(*e) > loud - 35.0).collect();
        if !act.is_empty() {
            let g = (self.target - db(act.iter().sum::<f64>() / act.len() as f64) as f32).clamp(-12.0, crate::MAX_GAIN_DB);
            let a = (-1.0f32 / 10.0).exp();
            self.gain_db = Some(match self.gain_db {
                // the first speech sets the gain at once; afterwards a lower gain is taken at once (louder speech than
                // the gain was set for: a quiet onset set it high, and the speech after it would come out tens of dB
                // too loud, clipped by the limiter, for the second the smoothing takes) and a higher one moves with
                // a 0.5 s time constant
                None => g,
                Some(p) if g < p => g,
                Some(p) if act.len() < 4 => p,
                Some(p) => a * p + (1.0 - a) * g,
            });
        }
        let k = 10f32.powf(self.gain_db.unwrap_or(0.0) / 20.0);
        f.iter_mut().for_each(|x| *x *= k);
    }

    /// leveled samples of `x` (whole 50 ms frames; the rest waits for the next call or `flush`)
    pub fn push(&mut self, x: &[f32]) -> Vec<f32> {
        self.carry.extend_from_slice(x);
        let whole = self.carry.len() / self.hop * self.hop;
        let mut out: Vec<f32> = self.carry.drain(..whole).collect();
        for f in out.chunks_mut(self.hop) {
            self.frame(f);
        }
        out
    }

    pub fn flush(&mut self) -> Vec<f32> {
        let mut out = std::mem::take(&mut self.carry);
        if !out.is_empty() {
            self.frame(&mut out);
        }
        out
    }
}

/// Where a streamed take's sound goes: each piece through the leveler (or the headroom) and the soft limiter
pub struct Out<'a> {
    pub level: Option<Leveler>,
    pub headroom_db: f32,
    pub sink: &'a mut dyn FnMut(&[f32]) -> Result<()>,
    pub all: Vec<f32>,
}

impl Out<'_> {
    pub fn send(&mut self, pcm: &[f32]) -> Result<()> {
        let mut v = match self.level.as_mut() {
            Some(l) => l.push(pcm),
            None => pcm.to_vec(),
        };
        crate::output(&mut v, if self.level.is_some() { 0.0 } else { self.headroom_db });
        self.all.extend_from_slice(&v);
        if v.is_empty() { Ok(()) } else { (self.sink)(&v) }
    }

    pub fn finish(&mut self) -> Result<()> {
        if let Some(l) = self.level.as_mut() {
            let mut v = l.flush();
            crate::output(&mut v, 0.0);
            self.all.extend_from_slice(&v);
            if !v.is_empty() {
                (self.sink)(&v)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_ramp_and_hold_one_frame_back() {
        let p = Prefix::new(3);
        assert_eq!(p.due(7, None), None);
        assert_eq!(p.due(8, None), Some((0, 3, 7, 8)));
        let p = Prefix { emitted: 200, np: 3 };
        assert_eq!(p.due(216, None), Some((0, 200, 215, 216)));
        assert_eq!(p.due(205, Some(205)), Some((0, 200, 205, 205)));
    }

    #[test]
    fn a_quiet_onset_does_not_make_the_speech_after_it_loud() {
        let rate = 44100usize;
        // 0.1 s at -60 dBFS, then 3 s at about -10 dBFS
        let x: Vec<f32> = (0..rate * 31 / 10).map(|i| (if i < rate / 10 { 0.001 } else { 0.45 }) * (i as f32 * 0.05).sin()).collect();
        let mut l = Leveler::new(rate as u32, -19.0);
        let mut y = l.push(&x);
        y.extend(l.flush());
        // every 50 ms of the loud part within 3 dB of the target
        for f in y[rate / 10 + rate / 20..].chunks(rate / 20).filter(|f| f.len() == rate / 20) {
            let db = 10.0 * (f.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / f.len() as f64).log10();
            assert!(db < -16.0, "{db}");
        }
    }

    #[test]
    fn the_causal_leveler_brings_a_quiet_tone_up() {
        let rate = 44100;
        let x: Vec<f32> = (0..rate * 4).map(|i| 0.01 * (i as f32 * 0.05).sin()).collect();
        let mut l = Leveler::new(rate as u32, -19.0);
        let mut y = l.push(&x);
        y.extend(l.flush());
        let tail = &y[y.len() - rate..];
        let db = 10.0 * (tail.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / tail.len() as f64).log10();
        assert!((db + 19.0).abs() < 1.0, "{db}");
    }
}
