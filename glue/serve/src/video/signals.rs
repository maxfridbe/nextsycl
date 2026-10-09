//! SIGTERM / SIGINT as a flag the daemon polls: `podman stop` then means the same as `h3 shutdown` - the running job
//! stops at its next block boundary and the engine unloads before the process exits.

use std::ffi::c_int;
use std::sync::atomic::{AtomicBool, Ordering};

static TERM: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(signum: c_int, handler: extern "C" fn(c_int)) -> usize;
}
const SIGINT: c_int = 2;
const SIGTERM: c_int = 15;

extern "C" fn on_signal(_: c_int) {
    TERM.store(true, Ordering::SeqCst); // an atomic store is all a signal handler may do here
}

pub fn install() {
    // SAFETY: the handler only stores to an atomic.
    unsafe {
        signal(SIGTERM, on_signal);
        signal(SIGINT, on_signal);
    }
}

pub fn terminated() -> bool {
    TERM.load(Ordering::SeqCst)
}
