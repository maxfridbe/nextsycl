//! NS_UP_PROFILE=N: each layer timed alone (a wait before and after it), with the floating-point operations and the
//! bytes its shapes imply, summed by kind; the table against the card's peaks every N pictures (a clip's frames;
//! default 50). The waits cost a little; the table is for finding where the time goes, not for the run's own speed.
//!
//! The peaks: Xe2's XMX does 2048 half (and bfloat16) FLOPs a clock on each Xe core (8 engines x 256; Intel's B580: 233 int8
//! TOPS from 20 Xe cores at 2.85 GHz, bfloat16 half that), so a card's is cores (compute units / 8) x 2048 x its
//! clock - the B70's 32 cores at 2.8 GHz: 183.5 TFLOPS. Its memory: 256 bits of 19 Gbps GDDR6, 608 GB/s; the B65's
//! the same bus. `image check --stages bench` measures what is reached: a large GEMM and a large copy.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use nextsycl_core::{Gpu, Result};

#[derive(Default, Clone, Copy)]
struct Stat {
    seconds: f64,
    flops: f64,
    bytes: f64,
    calls: u64,
}

static ON: OnceLock<bool> = OnceLock::new();
static STATS: Mutex<BTreeMap<String, Stat>> = Mutex::new(BTreeMap::new());

pub fn on() -> bool {
    *ON.get_or_init(|| std::env::var_os("NS_UP_PROFILE").is_some())
}

/// `f` timed alone as `what` (its FLOPs and bytes moved), when profiling; else just run
pub fn timed<T>(gpu: &Gpu, what: &str, flops: f64, bytes: f64, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !on() {
        return f();
    }
    gpu.sync()?;
    let t0 = Instant::now();
    let r = f()?;
    gpu.sync()?;
    let mut s = STATS.lock().unwrap();
    let e = s.entry(what.to_string()).or_default();
    e.seconds += t0.elapsed().as_secs_f64();
    e.flops += flops;
    e.bytes += bytes;
    e.calls += 1;
    Ok(r)
}

/// The card's bfloat16 XMX peak (TFLOPS) and memory bandwidth (GB/s)
pub fn peaks(gpu: &Gpu) -> (f64, f64) {
    let (cu, mhz) = nextsycl_core::gpu_units(gpu.index).unwrap_or((0, 0));
    (cu as f64 / 8.0 * 2048.0 * mhz as f64 * 1e6 / 1e12, 608.0)
}

/// The table since the last one (then cleared): each kind's seconds, share, TFLOPS and GB/s against the peaks
pub fn table(gpu: &Gpu) -> Vec<String> {
    let mut s = STATS.lock().unwrap();
    if s.is_empty() {
        return Vec::new();
    }
    let (pf, pb) = peaks(gpu);
    let total: f64 = s.values().map(|v| v.seconds).sum();
    let mut rows: Vec<(String, Stat)> = s.iter().map(|(k, v)| (k.clone(), *v)).collect();
    rows.sort_by(|a, b| b.1.seconds.total_cmp(&a.1.seconds));
    let mut out = vec![format!("upscale profile on {} (peaks {pf:.1} TFLOPS half, {pb:.0} GB/s): {total:.2} s timed", gpu.name),
                       format!("  {:<44} {:>6} {:>8} {:>6} {:>8} {:>6} {:>8} {:>6}", "step", "calls", "seconds", "share", "TFLOPS", "%peak", "GB/s", "%peak")];
    for (k, v) in rows {
        let (tf, gb) = (v.flops / v.seconds.max(1e-9) / 1e12, v.bytes / v.seconds.max(1e-9) / 1e9);
        out.push(format!("  {k:<44} {:>6} {:>8.3} {:>5.1}% {:>8.2} {:>5.1}% {:>8.1} {:>5.1}%", v.calls, v.seconds, 100.0 * v.seconds / total.max(1e-9), tf,
                         100.0 * tf / pf.max(1e-9), gb, 100.0 * gb / pb));
    }
    s.clear();
    out
}

/// After a picture: the table printed every NS_UP_PROFILE pictures (default 50)
pub fn picture_done(gpu: &Gpu) {
    if !on() {
        return;
    }
    static N: Mutex<u64> = Mutex::new(0);
    let every = std::env::var("NS_UP_PROFILE").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|n| *n > 0).unwrap_or(50);
    let mut n = N.lock().unwrap();
    *n += 1;
    if n.is_multiple_of(every) {
        for line in table(gpu) {
            eprintln!("{line}");
        }
    }
}
