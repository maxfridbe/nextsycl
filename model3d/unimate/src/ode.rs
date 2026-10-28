//! torchdiffeq 0.2.5's `odeint(method="dopri5")` as UniMate's sampler calls it (atol 1e-6, rtol 1e-3, the state float32,
//! time float64), transcribed from its source: the Dormand-Prince-Shampine tableau cast to float32, the initial step
//! of Hairer's recipe, the step-size controller (safety 0.9, ifactor 10, dfactor 0.2), steps that are not cut at the
//! end time, and the 4th order interpolant that gives the value at t = 1 from the step that passed it.
//!
//! The error norm and the tolerance are float64 as torch computes them (atol is a one-element float64 tensor in the
//! reference's call, which promotes the tolerance); the sums over the stages are float32 and left to right.

use nextsycl_model3d::Result;

/// The derivative: (time as float32, state) -> d state / dt
pub type Drift<'a> = dyn FnMut(f32, &[f32]) -> Result<Vec<f32>> + 'a;

const ALPHA: [f64; 6] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

fn beta() -> [Vec<f64>; 6] {
    [
        vec![1.0 / 5.0],
        vec![3.0 / 40.0, 9.0 / 40.0],
        vec![44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        vec![19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
        vec![9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
        vec![35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
    ]
}

fn c_error() -> [f64; 7] {
    [35.0 / 384.0 - 1951.0 / 21600.0, 0.0, 500.0 / 1113.0 - 22642.0 / 50085.0, 125.0 / 192.0 - 451.0 / 720.0, -2187.0 / 6784.0 - -12231.0 / 42400.0,
     11.0 / 84.0 - 649.0 / 6300.0, -1.0 / 60.0]
}

fn mid() -> [f64; 7] {
    [6025192743.0 / 30085553152.0 / 2.0, 0.0, 51252292925.0 / 65400821598.0 / 2.0, -2691868925.0 / 45128329728.0 / 2.0,
     187940372067.0 / 1594534317056.0 / 2.0, -1776094331.0 / 19743644256.0 / 2.0, 11237099.0 / 235043384.0 / 2.0]
}

/// What the solver did: evaluations of the drift, and each step (start, end, accepted)
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub nfe: usize,
    pub steps: Vec<(f64, f64, bool)>,
}

/// sum_j k[j][n] * c[j], float32, left to right
fn combine(k: &[Vec<f32>], c: &[f32], n: usize) -> f32 {
    let mut s = k[0][n] * c[0];
    for j in 1..c.len() {
        s += k[j][n] * c[j];
    }
    s
}

/// rms in float64 of err / (atol + rtol max(|a|, |b|)) with the product rtol max taken in float32
fn error_ratio(err: &[f32], a: &[f32], b: &[f32], rtol: f64, atol: f64) -> f64 {
    let r = rtol as f32;
    let mut s = 0f64;
    for i in 0..err.len() {
        let tol = atol + (r * a[i].abs().max(b[i].abs())) as f64;
        let e = err[i] as f64 / tol;
        s += e * e;
    }
    (s / err.len() as f64).sqrt()
}

fn rms(x: impl Iterator<Item = f64>, n: usize) -> f64 {
    (x.map(|v| v * v).sum::<f64>() / n as f64).sqrt()
}

/// y(t_end) from y(0) = y0: the state float32, the time float64
pub fn dopri5(f: &mut Drift, y0: &[f32], t_end: f64, rtol: f64, atol: f64) -> Result<(Vec<f32>, Stats)> {
    let n = y0.len();
    let mut st = Stats::default();
    let mut call = |t: f32, y: &[f32], st: &mut Stats| -> Result<Vec<f32>> {
        st.nfe += 1;
        f(t, y)
    };
    let f0 = call(0.0, y0, &mut st)?;

    // _select_initial_step (order 4: the exponent 1 / 5)
    let r32 = rtol as f32;
    let scale: Vec<f64> = y0.iter().map(|y| atol + (y.abs() * r32) as f64).collect();
    let d0 = rms(y0.iter().zip(&scale).map(|(y, s)| *y as f64 / s), n);
    let d1 = rms(f0.iter().zip(&scale).map(|(y, s)| *y as f64 / s), n);
    let h0 = if d0 < 1e-5 || d1 < 1e-5 { 1e-6f64 } else { 0.01 * d0 / d1 }.abs();
    let h0f = h0 as f32;
    let ya: Vec<f32> = y0.iter().zip(&f0).map(|(y, f)| y + h0f * f).collect();
    let f1 = call(h0 as f32, &ya, &mut st)?;
    let d2 = (rms(f1.iter().zip(&f0).zip(&scale).map(|((a, b), s)| (a - b) as f64 / s), n) / h0).abs();
    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 { (h0 * 1e-3).max(1e-6) } else { (0.01 / d1.max(d2)).powf(1.0 / 5.0) }.abs();
    let mut dt = (100.0 * h0).min(h1);

    let alpha: Vec<f32> = ALPHA.iter().map(|a| *a as f32).collect();
    let beta: Vec<Vec<f32>> = beta().iter().map(|b| b.iter().map(|x| *x as f32).collect()).collect();
    let cerr: Vec<f32> = c_error().iter().map(|x| *x as f32).collect();
    let cmid: Vec<f32> = mid().iter().map(|x| *x as f32).collect();

    let (mut y, mut fcur) = (y0.to_vec(), f0);
    let mut t1 = 0f64;
    let mut coeff: Vec<Vec<f32>> = Vec::new();
    let mut last_t0 = 0f64;
    while t_end > t1 {
        let t0 = t1;
        if !dt.is_finite() || t0 + dt <= t0 {
            return Err(nextsycl_model3d::Error(format!("dopri5: the step size underflowed at t = {t0} (dt {dt})")));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(nextsycl_model3d::Error("dopri5: non-finite values in the state".into()));
        }
        let (t0f, dtf, t1new) = (t0 as f32, dt as f32, t0 + dt);
        let t1f = t1new as f32;
        let mut k: Vec<Vec<f32>> = vec![fcur.clone()];
        for i in 0..6 {
            let ti = if ALPHA[i] == 1.0 { t1f } else { t0f + alpha[i] * dtf };
            let bd: Vec<f32> = beta[i].iter().map(|b| b * dtf).collect();
            let yi: Vec<f32> = (0..n).map(|j| y[j] + combine(&k, &bd, j)).collect();
            k.push(call(ti, &yi, &mut st)?);
        }
        // the last stage's point is the solution (Dormand-Prince: the stage-6 coefficients are the solution's)
        let bd: Vec<f32> = beta[5].iter().map(|b| b * dtf).collect();
        let y1: Vec<f32> = (0..n).map(|j| y[j] + combine(&k, &bd, j)).collect();
        let ce: Vec<f32> = cerr.iter().map(|c| dtf * c).collect();
        let err: Vec<f32> = (0..n).map(|j| combine(&k, &ce, j)).collect();
        let ratio = error_ratio(&err, &y, &y1, rtol, atol);
        let accept = ratio <= 1.0;
        st.steps.push((t0, t1new, accept));
        if accept {
            let cm: Vec<f32> = cmid.iter().map(|c| dtf * c).collect();
            let ymid: Vec<f32> = (0..n).map(|j| y[j] + combine(&k, &cm, j)).collect();
            let (f0k, f1k) = (&k[0], &k[6]);
            let mut c: Vec<Vec<f32>> = (0..5).map(|_| Vec::with_capacity(n)).collect();
            for j in 0..n {
                let (y0j, y1j, ym, a0, a1) = (y[j], y1[j], ymid[j], f0k[j], f1k[j]);
                c[4].push(2.0 * dtf * (a1 - a0) - 8.0 * (y1j + y0j) + 16.0 * ym);
                c[3].push(dtf * (5.0 * a0 - 3.0 * a1) + 18.0 * y0j + 14.0 * y1j - 32.0 * ym);
                c[2].push(dtf * (a1 - 4.0 * a0) - 11.0 * y0j - 5.0 * y1j + 16.0 * ym);
                c[1].push(dtf * a0);
                c[0].push(y0j);
            }
            coeff = c;
            last_t0 = t0;
            t1 = t1new;
            fcur = k.pop().unwrap_or_default();
            y = y1;
        }
        // _optimal_step_size
        dt = if ratio == 0.0 {
            dt * 10.0
        } else {
            let dfactor = if ratio < 1.0 { 1.0 } else { 0.2 };
            dt * (10.0f64).min((0.9 / ratio.powf(1.0 / 5.0)).max(dfactor))
        };
    }
    // _interp_evaluate at t_end
    let x = ((t_end - last_t0) / (t1 - last_t0)) as f32;
    let out = (0..n).map(|j| {
        let mut total = coeff[0][j] + x * coeff[1][j];
        let mut xp = x;
        for c in &coeff[2..] {
            xp *= x;
            total += xp * c[j];
        }
        total
    }).collect();
    Ok((out, st))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decay_is_integrated_to_its_tolerance() {
        let mut f = |_: f32, y: &[f32]| -> Result<Vec<f32>> { Ok(y.iter().map(|v| -2.0 * v).collect()) };
        let (y, st) = dopri5(&mut f, &[1.0, 2.0], 1.0, 1e-3, 1e-6).unwrap();
        let want = (-2.0f64).exp();
        assert!((y[0] as f64 - want).abs() < 2e-3 * want, "{} vs {want}", y[0]);
        assert!((y[1] as f64 - 2.0 * want).abs() < 4e-3 * want);
        // 2 evaluations to start, 6 a step
        assert_eq!((st.nfe - 2) % 6, 0);
        assert!(st.nfe >= 14);
    }

    #[test]
    fn the_end_is_interpolated_not_stepped_to() {
        let mut f = |_: f32, y: &[f32]| -> Result<Vec<f32>> { Ok(vec![1.0; y.len()]) };
        let (y, st) = dopri5(&mut f, &[0.0], 1.0, 1e-3, 1e-6).unwrap();
        assert!((y[0] - 1.0).abs() < 1e-5);
        let last = st.steps.last().unwrap();
        assert!(last.1 >= 1.0);
    }
}
