//! Where a step's time goes (NS_HV15_PROFILE=1): the queue drained at each mark and the time since the last mark added
//! to that mark's name - device time, a little slower than a step without it. Off: a mark costs nothing.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

use nextsycl_core::Result;
use nextsycl_diffusion::kernels::Nsd;

pub struct Prof {
    pub on: bool,
    last: Mutex<Instant>,
    acc: Mutex<BTreeMap<&'static str, (f64, u64)>>,
}

impl Prof {
    pub fn new(on: bool) -> Prof {
        Prof { on, last: Mutex::new(Instant::now()), acc: Mutex::new(BTreeMap::new()) }
    }

    /// The clock from now (the work before it not counted)
    pub fn start(&self, nsd: &Nsd) -> Result<()> {
        if self.on {
            nsd.wait()?;
            if let Ok(mut l) = self.last.lock() {
                *l = Instant::now();
            }
        }
        Ok(())
    }

    /// The work queued since the last mark, as `name`
    pub fn mark(&self, nsd: &Nsd, name: &'static str) -> Result<()> {
        if !self.on {
            return Ok(());
        }
        nsd.wait()?;
        let now = Instant::now();
        if let (Ok(mut l), Ok(mut a)) = (self.last.lock(), self.acc.lock()) {
            let e = a.entry(name).or_insert((0.0, 0));
            e.0 += now.duration_since(*l).as_secs_f64();
            e.1 += 1;
            *l = now;
        }
        Ok(())
    }

    /// The totals since the last report, largest first, `per` passes: one line each, and they are cleared
    pub fn report(&self, per: usize) -> Vec<String> {
        let Ok(mut a) = self.acc.lock() else { return Vec::new() };
        let total: f64 = a.values().map(|v| v.0).sum();
        let mut v: Vec<(&&str, &(f64, u64))> = a.iter().collect();
        v.sort_by(|x, y| y.1 .0.total_cmp(&x.1 .0));
        let per = per.max(1) as f64;
        let mut out: Vec<String> = v.iter().map(|(n, (s, c))| format!("[profile {n:<16} {:8.3} s a pass  {:5.1}%  {} calls]", s / per, 100.0 * s / total.max(1e-12), c)).collect();
        out.push(format!("[profile {:<16} {:8.3} s a pass]", "all", total / per));
        a.clear();
        out
    }
}
