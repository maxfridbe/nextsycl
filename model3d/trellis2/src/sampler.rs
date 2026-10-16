//! TRELLIS.2's sampler, FlowEulerGuidanceIntervalSampler, on the host: rectified-flow Euler over t from 1 to 0 warped
//! by rescale_t (t' = r t / (1 + (r - 1) t)), classifier-free guidance only inside the guidance interval, the guided
//! x0 rescaled toward the conditioned one's standard deviation. The velocities come from the model (`v(x, 1000 t,
//! cond?)`); the arithmetic is the reference's, in float32 with its scalars cast as torch casts them.

use nextsycl_core::Result;

/// One stage's sampler settings (pipeline.json's `params`)
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub steps: u32,
    pub guidance: f64,
    pub rescale: f64,
    pub interval: (f64, f64),
    pub rescale_t: f64,
}

pub const SIGMA_MIN: f64 = 1e-5;

impl Params {
    pub fn from_json(v: &serde_json::Value) -> Params {
        let f = |k: &str, d: f64| v[k].as_f64().unwrap_or(d);
        let iv = v["guidance_interval"].as_array();
        Params {
            steps: v["steps"].as_u64().unwrap_or(12) as u32,
            guidance: f("guidance_strength", 7.5),
            rescale: f("guidance_rescale", 0.0),
            interval: (iv.and_then(|a| a.first()).and_then(|x| x.as_f64()).unwrap_or(0.0),
                       iv.and_then(|a| a.get(1)).and_then(|x| x.as_f64()).unwrap_or(1.0)),
            rescale_t: f("rescale_t", 1.0),
        }
    }

    /// The calls a run takes: two a guided step, one otherwise
    pub fn calls(&self) -> u32 {
        schedule(self.steps, self.rescale_t).iter().map(|(t, _)| if self.guided(*t) { 2 } else { 1 }).sum()
    }

    fn guided(&self, t: f64) -> bool {
        // Python floats: the interval and the warped t compared in double precision (t = 1/3 warped by 3 is 0.6000000000000001)
        self.guidance != 1.0 && self.interval.0 <= t && t <= self.interval.1
    }
}

/// The (t, t_prev) pairs: linspace(1, 0, steps + 1) warped
pub fn schedule(steps: u32, rescale_t: f64) -> Vec<(f64, f64)> {
    let n = steps as usize;
    let t: Vec<f64> = (0..=n).map(|i| {
        // numpy's linspace: start + i * step, the last exactly the stop
        let v = if i == n { 0.0 } else { 1.0 + i as f64 * (-1.0 / n as f64) };
        rescale_t * v / (1.0 + (rescale_t - 1.0) * v)
    }).collect();
    (0..n).map(|i| (t[i], t[i + 1])).collect()
}

/// torch's std (unbiased) over all values
fn std(x: &[f32]) -> f32 {
    let n = x.len() as f64;
    let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    var.sqrt() as f32
}

/// The sampler over `x` (the noise), `v(x, 1000 t, cond)` the model (cond false: the negative condition); `step(i, of)`
/// after each step. Returns the sample.
pub fn sample(mut x: Vec<f32>, p: &Params, mut v: impl FnMut(&[f32], f32, bool) -> Result<Vec<f32>>, mut step: impl FnMut(u32, u32) -> Result<()>)
              -> Result<Vec<f32>> {
    let sched = schedule(p.steps, p.rescale_t);
    let sm = SIGMA_MIN;
    for (i, (t, tp)) in sched.iter().copied().enumerate() {
        let tt = (1000.0 * t) as f32;
        let g = if p.guided(t) { p.guidance } else { 1.0 };
        let pred = if g == 1.0 {
            v(&x, tt, true)?
        } else if g == 0.0 {
            v(&x, tt, false)?
        } else {
            let pos = v(&x, tt, true)?;
            let neg = v(&x, tt, false)?;
            let (a, b) = (g as f32, (1.0 - g) as f32);
            let mut pred: Vec<f32> = pos.iter().zip(&neg).map(|(p, n)| a * p + b * n).collect();
            if p.rescale > 0.0 {
                let (c1, c2) = ((1.0 - sm) as f32, (sm + (1.0 - sm) * t) as f32);
                let x0p: Vec<f32> = x.iter().zip(&pos).map(|(x, v)| c1 * x - c2 * v).collect();
                let x0c: Vec<f32> = x.iter().zip(&pred).map(|(x, v)| c1 * x - c2 * v).collect();
                let k = std(&x0p) / std(&x0c);
                let (r, r1) = (p.rescale as f32, (1.0 - p.rescale) as f32);
                pred = x.iter().zip(&x0c).map(|(x, c)| {
                    let x0 = r * (c * k) + r1 * c;
                    (c1 * x - x0) / c2
                }).collect();
            }
            pred
        };
        let dt = (t - tp) as f32;
        x.iter_mut().zip(&pred).for_each(|(x, v)| *x -= dt * v);
        step(i as u32 + 1, p.steps)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_is_warped_linspace() {
        let s = schedule(12, 5.0);
        assert_eq!(s.len(), 12);
        assert_eq!(s[0].0, 1.0);
        assert_eq!(s[11].1, 0.0);
        // t = 0.5 warped by 5: 2.5 / 3
        assert!((s[6].0 - 2.5 / 3.0).abs() < 1e-12);
        assert!(s.windows(2).all(|w| w[0].0 > w[1].0));
    }

    #[test]
    fn guided_steps_are_counted() {
        // TRELLIS.2's sparse structure: 10 of 12 steps inside [0.6, 1] -> 22 calls; the texture's, unguided: 12
        let ss = Params { steps: 12, guidance: 7.5, rescale: 0.7, interval: (0.6, 1.0), rescale_t: 5.0 };
        assert_eq!(ss.calls(), 22);
        let tex = Params { steps: 12, guidance: 1.0, rescale: 0.0, interval: (0.6, 0.9), rescale_t: 3.0 };
        assert_eq!(tex.calls(), 12);
        // the shape's: t = 1/3 warped by 3 lands on 0.6000000000000001, inside -> 9 guided steps, 21 calls (the reference's)
        let shape = Params { steps: 12, guidance: 7.5, rescale: 0.5, interval: (0.6, 1.0), rescale_t: 3.0 };
        assert_eq!(shape.calls(), 21);
    }

    #[test]
    fn guidance_and_rescale_follow_the_reference() {
        // a model whose conditioned velocity is x + 1 and unconditioned x - 1: one step at t = 1 -> 0
        let p = Params { steps: 1, guidance: 3.0, rescale: 0.5, interval: (0.0, 1.0), rescale_t: 1.0 };
        let x0 = vec![0.5f32, -1.0, 2.0];
        let out = sample(x0.clone(), &p, |x, _, c| Ok(x.iter().map(|v| if c { v + 1.0 } else { v - 1.0 }).collect()), |_, _| Ok(())).unwrap();
        // by hand: pred = 3 pos - 2 neg = x + 5; x0 = (1 - s) x - (s + (1 - s)) pred
        let sm = SIGMA_MIN as f32;
        let (c1, c2) = (1.0 - sm, 1.0f32);
        let pos: Vec<f32> = x0.iter().map(|v| v + 1.0).collect();
        let cfg: Vec<f32> = x0.iter().map(|v| v + 5.0).collect();
        let x0p: Vec<f32> = x0.iter().zip(&pos).map(|(x, v)| c1 * x - c2 * v).collect();
        let x0c: Vec<f32> = x0.iter().zip(&cfg).map(|(x, v)| c1 * x - c2 * v).collect();
        let k = std(&x0p) / std(&x0c);
        for i in 0..3 {
            let z = 0.5 * x0c[i] * k + 0.5 * x0c[i];
            let pred = (c1 * x0[i] - z) / c2;
            assert!((out[i] - (x0[i] - pred)).abs() < 1e-5, "{i}: {} vs {}", out[i], x0[i] - pred);
        }
    }
}
