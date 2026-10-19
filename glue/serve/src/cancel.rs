//! Stopping one request of a server: a request may carry an id (`request_id`, the front door's job), and
//! `POST /api/cancel {"id"}` stops that request only - between steps when it runs, before it starts when it waits.
//! Without an id the cancel stops whatever runs (the servers' own pages). A server that runs one request at a time
//! asks `stopping()`; one that runs several together (an audio engine that batches) asks each request's
//! `Running::stopping()`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct Cancel {
    flag: AtomicBool,
    /// the requests running: each one's id, if it gave one, and its own stop
    current: Mutex<Vec<(Option<String>, Arc<AtomicBool>)>>,
    /// waiting requests cancelled before they started (dropped when they come up)
    dropped: Mutex<Vec<String>>,
}

impl Cancel {
    /// A request starts (it holds the engine): none when it was cancelled while it waited; the guard ends it
    pub fn begin(&self, id: Option<&str>) -> Option<Running<'_>> {
        if let Some(id) = id {
            let mut d = self.dropped.lock().unwrap();
            if let Some(i) = d.iter().position(|x| x == id) {
                d.remove(i);
                return None;
            }
        }
        let mut cur = self.current.lock().unwrap();
        // the first of those running: a stop meant for the ones before it is spent
        if cur.is_empty() {
            self.flag.store(false, Ordering::Relaxed);
        }
        let mine = Arc::new(AtomicBool::new(false));
        cur.push((id.map(str::to_string), mine.clone()));
        Some(Running(self, mine))
    }

    /// Whether the running request is to stop (a server that runs one at a time)
    pub fn stopping(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Stop the request `id` (running: at its next step; waiting: when it comes up), or with none whatever runs;
    /// whether it was running
    pub fn request(&self, id: Option<&str>, busy: bool) -> bool {
        let cur = self.current.lock().unwrap();
        match id {
            Some(id) if cur.iter().any(|(x, _)| x.as_deref() == Some(id)) => {
                for (_, f) in cur.iter().filter(|(x, _)| x.as_deref() == Some(id)) {
                    f.store(true, Ordering::Relaxed);
                }
                // one at a time: the server's own stop is this request's
                if cur.len() == 1 {
                    self.flag.store(true, Ordering::Relaxed);
                }
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
                for (_, f) in cur.iter() {
                    f.store(true, Ordering::Relaxed);
                }
                true
            }
            None => false,
        }
    }

    /// The id of the request running (the first, when several run)
    pub fn current(&self) -> Option<String> {
        self.current.lock().unwrap().iter().find_map(|(id, _)| id.clone())
    }

    /// The requests running
    pub fn running(&self) -> usize {
        self.current.lock().unwrap().len()
    }
}

/// The request running: ended when dropped
pub struct Running<'a>(&'a Cancel, Arc<AtomicBool>);

impl Running<'_> {
    /// Whether this request is to stop
    pub fn stopping(&self) -> bool {
        self.1.load(Ordering::Relaxed)
    }
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        let mut cur = self.0.current.lock().unwrap();
        cur.retain(|(_, f)| !Arc::ptr_eq(f, &self.1));
        if cur.is_empty() {
            self.0.flag.store(false, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Cancel;

    #[test]
    fn a_cancel_stops_its_own_request_of_several() {
        let c = Cancel::default();
        let a = c.begin(Some("a")).unwrap();
        let b = c.begin(Some("b")).unwrap();
        assert_eq!(c.running(), 2);
        assert!(c.request(Some("b"), true));
        assert!(b.stopping() && !a.stopping());
        // several running: the one-at-a-time stop is not set by one of them
        assert!(!c.stopping());
        drop(b);
        // a waiting one cancelled is dropped when it comes up
        assert!(!c.request(Some("w"), true));
        assert!(c.begin(Some("w")).is_none());
        // no id: everything running
        assert!(c.request(None, true));
        assert!(a.stopping() && c.stopping());
        drop(a);
        assert_eq!(c.running(), 0);
        let one = c.begin(None).unwrap();
        assert!(!c.stopping() && !one.stopping());
    }
}
