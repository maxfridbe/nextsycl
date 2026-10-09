//! NS_QI_PROFILE=1: the GPU waits after each named step and the time is added to that name; `report` prints the
//! totals (slower overall - the waits break the queue's overlap - but it shows where the time goes).

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use nextsycl_core::Result;
use nextsycl_diffusion::kernels::Nsd;

struct State {
    last: Instant,
    t: BTreeMap<&'static str, (f64, u64)>,
}

fn state() -> Option<&'static Mutex<State>> {
    static S: OnceLock<Option<Mutex<State>>> = OnceLock::new();
    S.get_or_init(|| std::env::var_os("NS_QI_PROFILE").map(|_| Mutex::new(State { last: Instant::now(), t: BTreeMap::new() }))).as_ref()
}

pub fn on() -> bool {
    state().is_some()
}

/// The work queued since the last mark, waited for and counted as `name`
pub fn mark(nsd: &Nsd, name: &'static str) -> Result<()> {
    if let Some(s) = state() {
        nsd.wait()?;
        let mut s = s.lock().unwrap();
        let dt = s.last.elapsed().as_secs_f64();
        let e = s.t.entry(name).or_default();
        e.0 += dt;
        e.1 += 1;
        s.last = Instant::now();
    }
    Ok(())
}

/// Start a new interval without counting the time before it
pub fn reset_clock(nsd: &Nsd) -> Result<()> {
    if let Some(s) = state() {
        nsd.wait()?;
        s.lock().unwrap().last = Instant::now();
    }
    Ok(())
}

/// The totals, largest first, and clear them
pub fn report(log: &mut dyn FnMut(String)) {
    if let Some(s) = state() {
        let mut s = s.lock().unwrap();
        let total: f64 = s.t.values().map(|v| v.0).sum();
        let mut v: Vec<_> = s.t.iter().map(|(k, v)| (*k, *v)).collect();
        v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        log(format!("profile: {total:.3} s"));
        for (k, (t, n)) in v {
            log(format!("  {k:<18} {:>8.1} ms  {:>5.1}%  {n:>6} calls  {:>8.3} ms/call", t * 1e3, 100.0 * t / total.max(1e-9), t * 1e3 / n as f64));
        }
        s.t.clear();
    }
}
