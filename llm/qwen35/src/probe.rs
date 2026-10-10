//! NS_Q35_PROFILE=1 (with NS_PROFILE=gpu: the queue keeps device timestamps): every kernel call of a pass timed on the
//! device between two stamps, with the bytes it moves and the arithmetic it does, so each part shows its time, its
//! bandwidth and its compute against the card's ceilings - and the pass's device time no kernel ran (the GPU waiting
//! for the host's launches). Decode passes (at most 8 rows) and prompt passes are reported apart.
//!
//! Bytes are what a kernel reads and writes once (weights, activations, caches); a kernel that reads the same data
//! from several work-groups (attention's keys in a prompt pass) shows the bytes it asks for, which the caches may
//! serve - so its "of the bandwidth" can pass 100%. Ceilings: NS_PEAK_GBS (608: the B70's GDDR6), NS_PEAK_TF32
//! (22.9: 256 vector engines x 16 lanes x 2 x 2.8 GHz), NS_PEAK_TF16 (183: XMX, half), for the half GEMMs.

use std::collections::BTreeMap;
use std::ffi::c_int;
use std::sync::Mutex;

use nextsycl_core::{Error, Result};

struct Rec {
    phase: &'static str,
    name: &'static str,
    key: String,
    t0: i64,
    t1: i64,
    bytes: u64,
    flops: u64,
}

#[derive(Default)]
struct Row {
    ns: f64,
    calls: u64,
    bytes: u64,
    flops: u64,
}

#[derive(Default)]
struct Mode {
    /// (phase, name, key) -> totals
    rows: BTreeMap<(&'static str, &'static str, String), Row>,
    /// the passes' device spans (first stamp to last), passes, rows (tokens)
    span_ns: f64,
    passes: u64,
    tokens: u64,
}

#[derive(Default)]
struct Inner {
    pending: Vec<Rec>,
    phase: &'static str,
    /// a stamp's own time (two in a row), taken off each kernel's
    stamp_ns: Option<f64>,
    modes: [Mode; 2],
}

pub struct Probe {
    gpu: *mut std::ffi::c_void,
    inner: Mutex<Inner>,
}

// SAFETY: the GPU handle is used under the engine's work lock, one pass at a time.
unsafe impl Send for Probe {}
unsafe impl Sync for Probe {}

fn peak(var: &str, def: f64) -> f64 {
    std::env::var(var).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}

impl Probe {
    /// The probe when NS_Q35_PROFILE=1; an error when the queue keeps no timestamps
    pub fn new(gpu: *mut std::ffi::c_void) -> Result<Option<Probe>> {
        if std::env::var("NS_Q35_PROFILE").is_ok_and(|v| v == "1") {
            if !std::env::var("NS_PROFILE").is_ok_and(|v| v == "gpu") {
                return Err(Error("NS_Q35_PROFILE=1 times kernels on the device: set NS_PROFILE=gpu too (the queue's timestamps)".into()));
            }
            return Ok(Some(Probe { gpu, inner: Mutex::new(Inner { phase: "", ..Default::default() }) }));
        }
        Ok(None)
    }

    fn stamp(&self) -> Result<i64> {
        let a = nextsycl_core::api()?;
        let mut t = 0;
        // SAFETY: an out-pointer to a local.
        if unsafe { (a.stamp)(self.gpu, &mut t) } != 0 {
            return Err(Error(format!("stamp: {}", a.error())));
        }
        Ok(t)
    }

    fn elapsed(&self, t0: i64, t1: i64) -> Result<f64> {
        let a = nextsycl_core::api()?;
        let mut ns = 0.0;
        // SAFETY: two tickets of this GPU's ring; an out-pointer to a local.
        if unsafe { (a.elapsed)(self.gpu, t0, t1, &mut ns) } != 0 {
            return Err(Error(format!("elapsed: {}", a.error())));
        }
        Ok(ns)
    }

    /// What the calls from here on are part of ("deltanet", "ffn" ...)
    pub fn phase(&self, p: &'static str) {
        self.inner.lock().unwrap().phase = p;
    }

    /// A kernel call between two stamps
    pub fn op(&self, name: &'static str, key: String, bytes: u64, flops: u64, f: impl FnOnce() -> c_int) -> Result<c_int> {
        let t0 = self.stamp()?;
        let rc = f();
        let t1 = self.stamp()?;
        let mut m = self.inner.lock().unwrap();
        let phase = m.phase;
        m.pending.push(Rec { phase, name, key, t0, t1, bytes, flops });
        Ok(rc)
    }

    /// The pass is over (its results read back, so every stamp is done): its records into the tables
    pub fn settle(&self, rows: usize) -> Result<()> {
        let mut m = self.inner.lock().unwrap();
        if m.stamp_ns.is_none() {
            let mut s = 0.0;
            for _ in 0..16 {
                let (a, b) = (self.stamp()?, self.stamp()?);
                nextsycl_core::api().and_then(|a| if unsafe { (a.sync)(self.gpu) } == 0 { Ok(()) } else { Err(Error(a.error())) })?;
                s += self.elapsed(a, b)?;
            }
            m.stamp_ns = Some(s / 16.0);
        }
        let st = m.stamp_ns.unwrap_or(0.0);
        let pending = std::mem::take(&mut m.pending);
        let (Some(first), Some(last)) = (pending.first(), pending.last()) else { return Ok(()) };
        let span = self.elapsed(first.t0, last.t1)?;
        let mode = &mut m.modes[usize::from(rows > 8)];
        mode.span_ns += span;
        mode.passes += 1;
        mode.tokens += rows as u64;
        for r in &pending {
            let ns = (self.elapsed(r.t0, r.t1)? - st).max(0.0);
            let e = mode.rows.entry((r.phase, r.name, r.key.clone())).or_default();
            e.ns += ns;
            e.calls += 1;
            e.bytes += r.bytes;
            e.flops += r.flops;
        }
        Ok(())
    }

    /// The tables: by phase, then each kernel (by time), per token
    pub fn lines(&self) -> Vec<String> {
        let m = self.inner.lock().unwrap();
        let (gbs, tf32, tf16) = (peak("NS_PEAK_GBS", 608.0), peak("NS_PEAK_TF32", 22.9), peak("NS_PEAK_TF16", 183.0));
        let mut v = Vec::new();
        for (i, mode) in m.modes.iter().enumerate() {
            if mode.passes == 0 {
                continue;
            }
            let tok = mode.tokens.max(1) as f64;
            let busy: f64 = mode.rows.values().map(|r| r.ns).sum();
            v.push(format!("[probe {}: {} passes, {} tokens; device {:.3} ms a token, kernels {:.3} ({:.1}%), between kernels {:.3} ({:.1}%); a stamp {:.1} us]",
                           if i == 0 { "decode" } else { "prompt" }, mode.passes, mode.tokens, mode.span_ns / tok / 1e6, busy / tok / 1e6,
                           100.0 * busy / mode.span_ns.max(1.0), (mode.span_ns - busy).max(0.0) / tok / 1e6,
                           100.0 * (mode.span_ns - busy).max(0.0) / mode.span_ns.max(1.0), m.stamp_ns.unwrap_or(0.0) / 1e3));
            let mut phases: BTreeMap<&str, (f64, u64, u64)> = BTreeMap::new();
            for ((p, _, _), r) in &mode.rows {
                let e = phases.entry(p).or_default();
                e.0 += r.ns;
                e.1 += r.bytes;
                e.2 += r.flops;
            }
            let mut ph: Vec<_> = phases.into_iter().collect();
            ph.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
            for (p, (ns, b, f)) in ph {
                v.push(format!("[probe   {p:<16} {:>8.3} ms/token {:>5.1}%  {:>6.0} GB/s {:>5.1}% of {gbs:.0}  {:>7.2} TFLOP/s]", ns / tok / 1e6,
                               100.0 * ns / busy.max(1.0), b as f64 / ns.max(1.0), 100.0 * b as f64 / ns.max(1.0) / gbs, f as f64 / ns.max(1.0) / 1e3));
            }
            let mut rows: Vec<_> = mode.rows.iter().collect();
            rows.sort_by(|a, b| b.1.ns.total_cmp(&a.1.ns));
            for ((p, n, k), r) in rows {
                let ns = r.ns.max(1.0);
                let tf = r.flops as f64 / ns / 1e3;
                let pk = if *n == "gemm f16" { tf16 } else { tf32 };
                v.push(format!("[probe     {:<10} {:<12} {:<28} {:>8.3} ms/token {:>5.1}% {:>7} calls {:>8.1} us  {:>6.0} GB/s {:>5.1}%  {:>7.2} TF {:>5.1}%]",
                               p, n, k, r.ns / tok / 1e6, 100.0 * r.ns / busy.max(1.0), r.calls, r.ns / r.calls.max(1) as f64 / 1e3,
                               r.bytes as f64 / ns, 100.0 * r.bytes as f64 / ns / gbs, tf, 100.0 * tf / pk));
            }
        }
        v
    }
}
