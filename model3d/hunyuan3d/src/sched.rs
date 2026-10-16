//! The samplers' noise levels, as Tencent's schedulers make them. Hunyuan3D's time runs backwards: sigma 0 is pure
//! noise and 1 the shape; the model is given sigma (the scheduler's timestep / 1000) and the step is
//! `x += (sigma_next - sigma) v`. Each list ends with the final 1.

/// FlowMatchEulerDiscreteScheduler as the 2.1 pipeline drives it: linspace(0, 1, steps), then 1. (Its last step
/// is empty - sigma 1 to 1 - and is skipped.)
pub fn flow_match(steps: usize) -> Vec<f32> {
    let steps = steps.max(1);
    let mut s: Vec<f32> = (0..steps).map(|i| if steps == 1 { 0.0 } else { (i as f64 / (steps - 1) as f64) as f32 }).collect();
    s.push(1.0);
    s
}

/// ConsistencyFlowMatchEulerDiscreteScheduler (1,000 training steps, `pcm` consistency steps): the training sigmas
/// linspace(0, 1, 1000) at Euler steps [0, r - 1, 2 r - 1, ...] (r = 1000 / pcm), `steps` of them taken evenly, then 1
pub fn consistency(steps: usize, pcm: usize) -> Vec<f32> {
    const TRAIN: usize = 1000;
    let steps = steps.clamp(1, pcm);
    let ratio = TRAIN / pcm;
    let euler: Vec<usize> = std::iter::once(0).chain((1..pcm).map(|i| ((i * ratio) as f64).round() as usize - 1)).collect();
    let mut s: Vec<f32> = (0..steps)
        .map(|i| {
            let at = (i as f64 * pcm as f64 / steps as f64).floor() as usize;
            (euler[at] as f64 / (TRAIN - 1) as f64) as f32
        })
        .collect();
    s.push(1.0);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_turbo_schedule_is_tencents() {
        // the reference's 5 steps: timesteps 0, 199.2, 399.4, 599.6, 799.8
        let s = consistency(5, 100);
        let want = [0.0, 199.0 / 999.0, 399.0 / 999.0, 599.0 / 999.0, 799.0 / 999.0, 1.0];
        assert_eq!(s.len(), 6);
        for (a, b) in s.iter().zip(want) {
            assert!((a - b).abs() < 1e-7, "{s:?}");
        }
    }

    #[test]
    fn flow_match_runs_zero_to_one() {
        let s = flow_match(50);
        assert_eq!(s.len(), 51);
        assert_eq!((s[0], s[49], s[50]), (0.0, 1.0, 1.0));
        assert!((s[1] - 1.0 / 49.0).abs() < 1e-7);
        assert!(s.windows(2).all(|w| w[1] >= w[0]));
    }
}
