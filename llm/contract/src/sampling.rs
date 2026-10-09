//! Sampling on the host from a row of logits: the distribution (temperature, top-p over the likeliest 256) and a
//! draw from it. The server and `generate` use these; an engine that draws on the GPU says so (`Sampler::device`).

/// A tiny deterministic generator for sampling (xorshift64*).
pub struct Rng(pub u64);
impl Rng {
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// The distribution sampling draws from: softmax(logits / temp) over the 256 likeliest, restricted to the smallest set
/// of top tokens holding top_p, normalized. None at temperature 0 (greedy).
pub fn dist(logits: &[f32], temp: f32, top_p: f32) -> Option<Vec<(u32, f32)>> {
    if temp <= 0.0 {
        return None;
    }
    // the 256 likeliest in order (higher logit first, then lower index - a stable sort's order): selected in linear
    // time, only they sorted - the whole vocabulary's sort (154,880) cost ~2.5 ms a token (decode at temperature 1.0
    // 20.6 tok/s vs greedy's 23.1)
    let by = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b));
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = 256.min(idx.len());
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, by);
        idx.truncate(k);
    }
    idx.sort_by(by);
    let mx = logits[idx[0]];
    let mut p: Vec<(u32, f32)> = idx.iter().map(|&i| (i as u32, ((logits[i] - mx) / temp).exp())).collect();
    let sum: f32 = p.iter().map(|x| x.1).sum();
    let mut acc = 0.0;
    let mut keep = p.len();
    for (n, x) in p.iter_mut().enumerate() {
        x.1 /= sum;
        acc += x.1;
        if acc >= top_p {
            keep = n + 1;
            break;
        }
    }
    p.truncate(keep);
    let total: f32 = p.iter().map(|x| x.1).sum();
    for x in p.iter_mut() {
        x.1 /= total;
    }
    Some(p)
}

/// Greedy at temperature 0; else a token drawn from `dist`.
pub fn sample(logits: &[f32], temp: f32, top_p: f32, rng: &mut Rng) -> u32 {
    match dist(logits, temp, top_p) {
        None => logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32),
        Some(p) => {
            let mut r = rng.next_f32();
            for (i, w) in &p {
                r -= w;
                if r <= 0.0 {
                    return *i;
                }
            }
            p.last().map_or(0, |x| x.0)
        }
    }
}

