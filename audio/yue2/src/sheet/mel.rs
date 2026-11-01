//! MERT-v2's frontend on the host (modeling_mert2.py `MERT2MelFrontend`, torchaudio's transforms as it builds them):
//!
//! ```text
//!   Spectrogram(n_fft 2048, win 2048, hop 240, power 2): centred frames (the signal reflect-padded by 1024 either side),
//!     the file's periodic Hann window, |rfft|^2 - 1 + samples / 240 frames
//!   MelScale(128, 24 kHz, 1025 bins): the file's filter bank (HTK, no norm) - mel = power . fb
//!   AmplitudeToDB(power, no top_db): 10 log10(max(mel, 1e-10))
//!   the last frame dropped, (mel - mel_mean) / max(mel_std, 1e-5)            -> [frames, 128], 100 a second
//! ```
//!
//! The transforms run in float32 upstream (autocast off); here the FFT is float64 (an iterative radix-2 transform of
//! the 2048 points) and the mel sums too, rounded to float32 at the end: closer to the exact values than either, and
//! the silence that pads a short window is exactly 0 both ways (so exactly -100 dB both ways). The frames are split
//! across the host's threads.

pub const N_FFT: usize = 2048;
pub const HOP: usize = 240;
pub const BINS: usize = N_FFT / 2 + 1;
pub const MELS: usize = 128;

/// The frontend's constants from the file: its window, filter bank and statistics
pub struct Frontend {
    window: Vec<f64>,
    /// the filter bank by mel bin: (the first frequency bin it covers, its weights)
    filters: Vec<(usize, Vec<f64>)>,
    mean: Vec<f32>,
    std: Vec<f32>,
    twiddle: Vec<(f64, f64)>,
    rev: Vec<usize>,
}

impl Frontend {
    /// `fb` [1025, 128] (torchaudio's layout), `window` [2048], `mean` / `std` [128]
    pub fn new(window: &[f32], fb: &[f32], mean: &[f32], std: &[f32]) -> Result<Frontend, String> {
        if window.len() != N_FFT || fb.len() != BINS * MELS || mean.len() != MELS || std.len() != MELS {
            return Err("the mel frontend's tensors are not MERT-v2's ([2048], [1025, 128], [128], [128])".into());
        }
        let filters = (0..MELS).map(|m| {
            let col: Vec<f64> = (0..BINS).map(|f| fb[f * MELS + m] as f64).collect();
            let a = col.iter().position(|v| *v != 0.0).unwrap_or(0);
            let b = col.iter().rposition(|v| *v != 0.0).map_or(a, |b| b + 1);
            (a, col[a..b].to_vec())
        }).collect();
        let bits = N_FFT.trailing_zeros();
        let rev = (0..N_FFT).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect();
        let twiddle = (0..N_FFT / 2).map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64;
            (a.cos(), a.sin())
        }).collect();
        Ok(Frontend { window: window.iter().map(|v| *v as f64).collect(), filters, mean: mean.to_vec(), std: std.iter().map(|s| s.max(1e-5)).collect(), twiddle, rev })
    }

    /// One frame's power spectrum (the 1025 bins) of `x` (2048 windowed samples)
    fn power(&self, x: &[f64], re: &mut [f64], im: &mut [f64], out: &mut [f64]) {
        for i in 0..N_FFT {
            re[self.rev[i]] = x[i];
            im[self.rev[i]] = 0.0;
        }
        let mut len = 2;
        while len <= N_FFT {
            let half = len / 2;
            let step = N_FFT / len;
            for s in (0..N_FFT).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = self.twiddle[k * step];
                    let (a, b) = (s + k, s + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len *= 2;
        }
        for k in 0..BINS {
            out[k] = re[k] * re[k] + im[k] * im[k];
        }
    }

    /// The normalized log-mel features of `wave` (24 kHz mono): [frames, 128], frames = samples / 240 (the last of
    /// the STFT's 1 + samples / 240 dropped)
    pub fn features(&self, wave: &[f32]) -> Vec<f32> {
        let n = wave.len();
        let frames = n / HOP;
        let pad = N_FFT / 2;
        // reflect padding (torch's "reflect": the edge sample not repeated)
        let at = |i: isize| -> f64 {
            let mut j = i - pad as isize;
            if j < 0 {
                j = -j;
            }
            if j >= n as isize {
                j = 2 * (n as isize - 1) - j;
            }
            wave[j.clamp(0, n as isize - 1) as usize] as f64
        };
        let mut out = vec![0f32; frames * MELS];
        let threads = std::thread::available_parallelism().map_or(4, |c| c.get()).clamp(1, 16);
        let per = frames.div_ceil(threads).max(1);
        std::thread::scope(|s| {
            for (ci, chunk) in out.chunks_mut(per * MELS).enumerate() {
                let at = &at;
                s.spawn(move || {
                    let (mut x, mut re, mut im, mut p) = (vec![0f64; N_FFT], vec![0f64; N_FFT], vec![0f64; N_FFT], vec![0f64; BINS]);
                    for (j, row) in chunk.chunks_mut(MELS).enumerate() {
                        let t = ci * per + j;
                        let s0 = (t * HOP) as isize;
                        for (i, v) in x.iter_mut().enumerate() {
                            *v = at(s0 + i as isize) * self.window[i];
                        }
                        self.power(&x, &mut re, &mut im, &mut p);
                        for (m, (a, w)) in self.filters.iter().enumerate() {
                            let e: f64 = w.iter().zip(&p[*a..]).map(|(w, p)| w * p).sum();
                            let db = 10.0 * e.max(1e-10).log10();
                            row[m] = ((db as f32) - self.mean[m]) / self.std[m];
                        }
                    }
                });
            }
        });
        out
    }
}
