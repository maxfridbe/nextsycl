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
