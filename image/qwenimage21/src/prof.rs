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

/// NS_QI_MEM=1: the card's free memory at named points (the GPU waited for first), the lowest per phase kept;
/// `mem_report` prints them. Measures what a size needs beside the loaded model.
/// Each phase's (lowest free bytes, probes)
type Lows = Mutex<BTreeMap<&'static str, (u64, u64)>>;

fn mem_state() -> Option<&'static Lows> {
    static S: OnceLock<Option<Lows>> = OnceLock::new();
    S.get_or_init(|| std::env::var_os("NS_QI_MEM").map(|_| Mutex::new(BTreeMap::new()))).as_ref()
}

pub fn mem_on() -> bool {
    mem_state().is_some()
}

/// The card's free bytes now (after its queue), counted toward `phase`'s lowest
pub fn mem(nsd: &Nsd, phase: &'static str) -> Result<()> {
    if let Some(s) = mem_state() {
        nsd.wait()?;
        nsd.gpu.sync()?;
        let (total, free) = nsd.gpu.memory()?;
        let free = free.unwrap_or(total);
        let mut s = s.lock().unwrap();
        let e = s.entry(phase).or_insert((u64::MAX, 0));
        // a new low by 64 MiB or more: said now (a run that fails later still shows how far it got); NS_QI_MEM=2: every probe
        if free + (64 << 20) < e.0 || std::env::var("NS_QI_MEM").is_ok_and(|v| v == "2") {
            eprintln!("mem: {phase} free {:.2} GiB", free as f64 / (1u64 << 30) as f64);
        }
        e.0 = e.0.min(free);
        e.1 += 1;
    }
    Ok(())
}

/// Each phase's lowest free memory, and clear them
pub fn mem_report(log: &mut dyn FnMut(String)) {
    if let Some(s) = mem_state() {
        let mut s = s.lock().unwrap();
        for (k, (low, n)) in s.iter() {
            log(format!("mem: {k:<14} lowest free {:>6.2} GiB ({n} probes)", *low as f64 / (1u64 << 30) as f64));
        }
        s.clear();
    }
}
