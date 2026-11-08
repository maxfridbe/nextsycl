//! The sampler's starting noise, value for value what the reference pipeline draws: `torch.manual_seed(seed)` and
//! `torch.randn(shape, dtype=float32, device="cpu")` for each stream in turn (video, then audio), from one generator.
//!
//! PyTorch's CPU generator is the 32-bit Mersenne Twister (mt19937). `randn` on a float32 tensor of 16 values or
//! more first fills the whole tensor with uniforms (24 random bits each), then turns every block of 16 into normals
//! with Box-Muller (the first 8 values give the radius, the last 8 the angle). A length that is not a multiple of 16
//! redraws 16 fresh uniforms over the last 16 values and transforms those again.

/// mt19937, 32-bit, with the standard seeding: what `torch.manual_seed` sets up.
pub struct Mt19937 {
    mt: [u32; 624],
    i: usize,
}

impl Mt19937 {
    pub fn new(seed: u64) -> Mt19937 {
        let mut mt = [0u32; 624];
        mt[0] = seed as u32;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, i: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// Uniform in [0, 1) from 24 random bits, as PyTorch makes a float32 uniform.
    fn uniform(&mut self) -> f32 {
        ((self.next_u32() & 0xff_ffff) as f64 * (1.0f64 / (1u64 << 24) as f64)) as f32
    }

    /// `torch.randn(n)` (float32, CPU) from this generator's current state.
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n).map(|_| self.uniform()).collect();
        if n < 16 {
            // PyTorch takes another path for tiny tensors; no latent is that small
            unimplemented!("randn of fewer than 16 values");
        }
        let mut i = 0;
        while i + 16 <= n {
            box_muller_16(&mut v[i..i + 16]);
            i += 16;
        }
        if !n.is_multiple_of(16) {
            let tail = &mut v[n - 16..];
            for x in tail.iter_mut() {
                *x = self.uniform();
            }
            box_muller_16(tail);
        }
        v
    }
}

fn box_muller_16(d: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0 - d[j];
        let u2 = d[j + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        d[j] = radius * theta.cos();
        d[j + 8] = radius * theta.sin();
    }
}

/// The starting noise of a clip: video [C, T, H, W] values, then audio [C, 2, T], drawn in that order.
pub fn clip_noise(seed: u64, video: usize, audio: usize) -> (Vec<f32>, Vec<f32>) {
    let mut g = Mt19937::new(seed);
    let v = g.randn(video);
    let a = g.randn(audio);
    (v, a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mersenne_twister_reference_values() {
        // std::mt19937 with the default seed 5489: the 10000th output is 4123659995 (C++ standard)
        let mut g = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10000 {
            last = g.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
    }

    #[test]
    fn randn_matches_torch() {
        // torch.manual_seed(0); torch.randn(4) is below 16 values (another path); torch.randn(16)[:4]:
        let mut g = Mt19937::new(0);
        let v = g.randn(16);
        let want = [-1.1258398, -1.1523602, -0.25057858, -0.4338788];
        for (a, b) in v.iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{:?}", &v[..4]);
        }
    }
}
