//! Stopping one request of a server that runs them one at a time: a request may carry an id (`request_id`, the
//! front door's job), and `POST /api/cancel {"id"}` stops that request only - between steps when it runs, before it
//! starts when it waits. Without an id the cancel stops whatever runs (the servers' own pages).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[derive(Default)]
pub struct Cancel {
    flag: AtomicBool,
    /// the id of the request running, if it gave one
    current: Mutex<Option<String>>,
    /// waiting requests cancelled before they started (dropped when they come up)
    dropped: Mutex<Vec<String>>,
}

impl Cancel {
    /// A request starts (it holds the engine): none when it was cancelled while it waited; the guard ends it
    pub fn begin(&self, id: Option<&str>) -> Option<Running<'_>> {
        self.flag.store(false, Ordering::Relaxed);
        if let Some(id) = id {
            let mut d = self.dropped.lock().unwrap();
            if let Some(i) = d.iter().position(|x| x == id) {
                d.remove(i);
                return None;
            }
        }
        *self.current.lock().unwrap() = id.map(str::to_string);
        Some(Running(self))
    }

    /// Whether the running request is to stop
    pub fn stopping(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Stop the request `id` (running: at its next step; waiting: when it comes up), or with none whatever runs;
    /// whether it was running
    pub fn request(&self, id: Option<&str>, busy: bool) -> bool {
        match id {
            Some(id) if self.current.lock().unwrap().as_deref() == Some(id) => {
                self.flag.store(true, Ordering::Relaxed);
                true
            }
            Some(id) => {
                let mut d = self.dropped.lock().unwrap();
                if !d.iter().any(|x| x == id) {
                    d.push(id.to_string());
                }
                // a bounded list: ids that never come up (answered elsewhere) age out
                let n = d.len();
                if n > 64 {
                    d.drain(..n - 64);
                }
                false
            }
            None if busy => {
                self.flag.store(true, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    pub fn current(&self) -> Option<String> {
        self.current.lock().unwrap().clone()
    }
}

/// The request running: ended when dropped
pub struct Running<'a>(&'a Cancel);

impl Drop for Running<'_> {
    fn drop(&mut self) {
        *self.0.current.lock().unwrap() = None;
        self.0.flag.store(false, Ordering::Relaxed);
    }
}
