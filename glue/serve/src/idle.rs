//! A server that lets its card go when nobody uses it (`--idle-exit SECONDS`): work - a generation, a page load -
//! marks it used; once it has been idle that long and nothing runs or waits, the process ends (between requests, so
//! never inside a kernel) and its memory goes back. `nextsycl serve` starts it again on the next request.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

pub struct Idle {
    last: AtomicU64,
    pub after: Option<Duration>,
}

impl Idle {
    pub fn new(after: Option<Duration>) -> Idle {
        Idle { last: AtomicU64::new(now()), after }
    }

    /// When work last happened (Unix seconds): the clock `after` runs from
    pub fn since(&self) -> u64 {
        self.last.load(Ordering::Relaxed)
    }

    /// Work happened
    pub fn touch(&self) {
        self.last.store(now(), Ordering::Relaxed);
    }

    /// Ends the process once idle `after` with `busy` false (checked every 10 s); nothing without `after`
    pub fn watch(&'static self, what: &'static str, busy: Box<dyn Fn() -> bool + Send>) {
        let Some(after) = self.after else { return };
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(10));
            if busy() {
                self.touch();
                continue;
            }
            if now().saturating_sub(self.last.load(Ordering::Relaxed)) >= after.as_secs() {
                eprintln!("{what}: idle for {} s - exiting (the card's memory goes back)", after.as_secs());
                std::process::exit(0);
            }
        });
    }
}

/// A server's /api/info with what the front door's timing fields read (home/timing.rs): when its model finished loading
/// (`loaded_at`), and its idle exit - the seconds it waits (`idle_exit_s`, null: never) and when work last happened
/// (`idle_since`)
pub fn stamp(info: &mut serde_json::Value, loaded_at: f64, idle: Option<&Idle>) {
    info["loaded_at"] = serde_json::json!(loaded_at);
    info["idle_exit_s"] = serde_json::json!(idle.and_then(|i| i.after).map(|d| d.as_secs()));
    info["idle_since"] = serde_json::json!(idle.filter(|i| i.after.is_some()).map(|i| i.since()));
}

/// Now, Unix seconds (float): a server's `loaded_at`
pub fn wall() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}
