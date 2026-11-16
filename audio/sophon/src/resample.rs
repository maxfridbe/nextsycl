//! A polyphase windowed-sinc resampler (the Whisper engine's: Hann, 32 zero crossings, cut at the lower Nyquist): the
//! client's audio to the ear's 16 kHz, the replies' 24 kHz to 16 kHz for the history.

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Mono samples at `rate` to `to`: a windowed sinc (Hann, 32 zero crossings, cut at the lower Nyquist), one table of
/// taps a phase when the ratio is a small fraction (every common rate to 16 kHz), else each sample's taps computed
pub fn resample(x: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || x.is_empty() || rate == 0 {
        return x.to_vec();
    }
    let g = gcd(rate as u64, to as u64);
    let (l, m) = ((to as u64 / g) as usize, (rate as u64 / g) as usize);
    let cut = (to as f64 / rate as f64).min(1.0);
    let zc = 32.0;
    let half = (zc / cut).ceil() as isize;
    let n = ((x.len() as u128 * l as u128) / m as u128) as usize;
    let tap = |d: f64| -> f64 {
        // d: the distance from the output's instant, in input samples, scaled by the cut
        let d = d * cut;
        if d.abs() >= zc {
            return 0.0;
        }
        let sinc = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d) };
        sinc * (0.5 + 0.5 * (std::f64::consts::PI * d / zc).cos()) * cut
    };
    let width = (2 * half + 1) as usize;
    // phase p: the output instant base + p / l; tap k at input base + k - half
    let table: Option<Vec<f32>> = (l <= 4096).then(|| {
        let mut t = vec![0f32; l * width];
        for p in 0..l {
            for k in 0..width {
                t[p * width + k] = tap((k as isize - half) as f64 - p as f64 / l as f64) as f32;
            }
        }
        t
    });
    let mut out = vec![0f32; n];
    let threads = std::thread::available_parallelism().map_or(4, |c| c.get()).clamp(1, 16);
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per).enumerate() {
            let table = table.as_deref();
            s.spawn(move || {
                for (j, o) in chunk.iter_mut().enumerate() {
                    let i = ci * per + j;
                    let num = i as u128 * m as u128;
                    let base = (num / l as u128) as isize;
                    let p = (num % l as u128) as usize;
                    let mut acc = 0f32;
                    for k in 0..width {
                        let at = base + k as isize - half;
                        if at < 0 || at as usize >= x.len() {
                            continue;
                        }
                        let w = match table {
                            Some(t) => t[p * width + k],
                            None => tap((k as isize - half) as f64 - p as f64 / l as f64) as f32,
                        };
                        acc += x[at as usize] * w;
                    }
                    *o = acc;
                }
            });
        }
    });
    out
}
