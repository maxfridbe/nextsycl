//! A song longer than the model's 300 s window, as pipeline_sheetsage2.py and generation_sheetsage2.py transcribe it:
//! windows every 100 s (`sliding_window_plan`: 300 s long, 200 s of overlap, the last 100 s of each but the last only
//! look-ahead), each window's decode started from the events already accepted in its overlap
//! (`build_overlap_prefix_tokens`: re-encoded with the window's own times, the context - structure, key, chord, meter -
//! carried into its first event) and stopped once a time token passes its acceptance boundary; its events placed in the
//! song by their subbeats through the timestamps (`event_time_map`) and kept inside [accept_start, accept_end)
//! (`stitched_window_events`).

use crate::sheet::tokens::{self, Decoded, Event, F_CHORD, F_KEY, F_RHYTHM, F_STRUCT, F_TIME, TIME0, TIME_HZ, N_TIME};

#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub start: f64,
    pub end: f64,
    pub accept_start: f64,
    pub accept_end: f64,
    pub prefix_end: f64,
    /// seconds into the window after which a time token ends the decode (None: the last window, which runs to the song's end)
    pub generation_stop: Option<f64>,
}

/// sliding_window_plan
pub fn plan(duration: f64, window: f64, overlap: f64, lookahead: f64) -> Result<Vec<Window>, String> {
    if duration.is_nan() || duration <= 0.0 || window.is_nan() || window <= 0.0 {
        return Err("Duration and window length must be positive".into());
    }
    if !(0.0 <= lookahead && lookahead <= overlap && overlap < window) {
        return Err("Require 0 <= lookahead <= overlap < window length".into());
    }
    let hop = window - overlap;
    let (mut start, mut accepted) = (0.0f64, 0.0f64);
    let mut out = Vec::new();
    loop {
        let last = start + window >= duration - 1e-6;
        let accept_end = if last { duration } else { start + window - lookahead };
        out.push(Window { start, end: duration.min(start + window), accept_start: accepted, accept_end, prefix_end: accepted,
                          generation_stop: if last { None } else { Some(window - lookahead) } });
        if last {
            return Ok(out);
        }
        accepted = accept_end;
        start = (start + hop).min(duration - window);
    }
}

/// np.interp
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let i = xs.partition_point(|v| *v <= x);
    if i == 0 {
        return ys[0];
    }
    if i >= xs.len() {
        return ys[ys.len() - 1];
    }
    let (x0, x1, y0, y1) = (xs[i - 1], xs[i], ys[i - 1], ys[i]);
    if x1 == x0 {
        return y1;
    }
    y0 + (x - x0) * ((y1 - y0) / (x1 - x0))
}

/// event_time_map: a subbeat's window-local seconds, through the timestamps' anchors (the median step beyond them)
pub struct TimeMap {
    steps: Vec<f64>,
    times: Vec<f64>,
    step_seconds: f64,
    target: f64,
}

impl TimeMap {
    pub fn new(d: &Decoded, target: f64) -> TimeMap {
        // dict(anchors): the last timestamp at a subbeat wins, sorted by subbeat
        let mut anchors: std::collections::BTreeMap<i64, f64> = std::collections::BTreeMap::new();
        for e in &d.events {
            if let Some(t) = e.timestamp() {
                anchors.insert(e.subbeat, t);
            }
        }
        let steps: Vec<f64> = anchors.keys().map(|k| *k as f64).collect();
        let times: Vec<f64> = anchors.values().copied().collect();
        let mut step_seconds = 0.125;
        if steps.len() >= 2 {
            let r: Vec<f64> = (1..steps.len()).map(|i| (times[i] - times[i - 1]) / (steps[i] - steps[i - 1]).max(1.0)).collect();
            let mut s = r.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            let n = s.len();
            let m = if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
            step_seconds = if m.is_finite() && m > 0.0 { m } else { 0.125 };
        }
        TimeMap { steps, times, step_seconds, target }
    }

    pub fn at(&self, step: f64) -> f64 {
        if self.steps.is_empty() {
            return self.target.min((step * 0.125).max(0.0));
        }
        let (s0, sn) = (self.steps[0], self.steps[self.steps.len() - 1]);
        if step <= s0 {
            return (self.times[0] + (step - s0) * self.step_seconds).clamp(0.0, self.target);
        }
        if step >= sn {
            return (self.times[self.times.len() - 1] + (step - sn) * self.step_seconds).clamp(0.0, self.target);
        }
        interp(step, &self.steps, &self.times)
    }
}

/// stitched_window_events: the window's events placed in the song, those in [accept_start, accept_end) kept
pub fn stitched(d: &Decoded, map: &TimeMap, w: &Window, duration: f64, index: usize, base: i64) -> Vec<Event> {
    let eps = 1e-4;
    let mut out = Vec::new();
    for e in &d.events {
        let abs = w.start + map.at(e.subbeat as f64);
        if abs < w.accept_start - eps || abs >= w.accept_end - eps || abs >= duration - eps {
            continue;
        }
        let mut o = e.clone();
        let time = abs.clamp(0.0, duration);
        o.time = Some(time);
        o.window_index = index;
        o.window_start = w.start;
        o.source_subbeat = e.subbeat;
        o.global_subbeat = base + e.subbeat;
        for n in &mut o.notes {
            let end = w.start + map.at((e.subbeat + n.duration_steps as i64) as f64);
            n.end_time = Some(duration.min((time + 0.04).max(end)));
        }
        out.push(o);
    }
    out
}

/// build_overlap_prefix_tokens: the next window's decode prefix from the accepted events in [window_start,
/// prefix_end), and the global subbeat its first event is at (None: no beat there to start from)
pub fn overlap_prefix(stitched: &[Event], prompts: &[u32], window_start: f64, prefix_end: f64) -> Result<Option<(Vec<u32>, i64)>, String> {
    let eps = 1e-4;
    let mut src: Vec<&Event> = stitched.iter().filter(|e| {
        let t = e.time.unwrap_or(-1.0);
        window_start - eps <= t && t < prefix_end - eps
    }).collect();
    src.sort_by(|a, b| a.global_subbeat.cmp(&b.global_subbeat).then(a.time.unwrap_or(0.0).total_cmp(&b.time.unwrap_or(0.0))));
    let Some(first) = src.iter().position(|e| e.has(F_TIME) || e.has(F_RHYTHM)) else { return Ok(None) };
    let src = &src[first..];
    let base = src[0].global_subbeat;
    // active_context_before: the structure, key, chord and meter in force at the first event
    let t0 = src[0].time.unwrap_or(0.0);
    let mut ctx: [Option<Vec<u32>>; 6] = Default::default();
    let mut meter: Option<Vec<u32>> = None;
    for e in stitched {
        let Some(t) = e.time else { continue };
        if t > t0 + 1e-6 {
            continue;
        }
        for f in [F_STRUCT, F_KEY, F_CHORD] {
            if e.has(f) {
                ctx[f] = Some(e.tokens[f].clone());
            }
        }
        let m: Vec<u32> = e.tokens[F_RHYTHM].iter().copied().filter(|t| tokens::ty(*t) == Some(tokens::Ty::Meter)).collect();
        if !m.is_empty() {
            meter = Some(m[..1].to_vec());
        }
    }
    let mut events = Vec::new();
    for s in src {
        let mut e = Event { subbeat: (s.global_subbeat - base).max(0), tokens: s.tokens.clone(), ..Default::default() };
        if e.has(F_TIME) {
            // set_event_local_timestamp: the time token re-made for the window
            let local = s.time.unwrap_or(0.0) - window_start;
            let id = ((local * TIME_HZ as f64).round_ties_even() as i64).clamp(0, N_TIME as i64 - 1);
            e.tokens[F_TIME] = vec![TIME0 + id as u32];
        }
        events.push(e);
    }
    // apply_prefix_context: the first event carries what is in force
    let e0 = &mut events[0];
    for f in [F_STRUCT, F_KEY, F_CHORD] {
        if !e0.has(f) {
            if let Some(c) = &ctx[f] {
                e0.tokens[f] = c.clone();
            }
        }
    }
    let r = &e0.tokens[F_RHYTHM];
    let has_meter = r.iter().any(|t| tokens::ty(*t) == Some(tokens::Ty::Meter));
    let has_eighth = r.iter().any(|t| tokens::ty(*t) == Some(tokens::Ty::Eighth));
    if has_eighth && !has_meter {
        if let Some(m) = &meter {
            let mut v = m.clone();
            v.extend_from_slice(r);
            e0.tokens[F_RHYTHM] = v;
        }
    }
    let d = Decoded { prompts: prompts.to_vec(), events, has_eos: false };
    Ok(Some((tokens::encode_decoded(&d)?, base)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_plan_is_upstreams() {
        let p = plan(132.816, 300.0, 200.0, 100.0).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].generation_stop, None);
        let p = plan(450.0, 300.0, 200.0, 100.0).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!((p[0].accept_end, p[1].start, p[1].accept_start, p[1].accept_end), (200.0, 100.0, 200.0, 300.0));
        assert_eq!((p[2].start, p[2].accept_end, p[2].generation_stop), (150.0, 450.0, None));
    }
}
