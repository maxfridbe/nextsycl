//! What the host (the box's front door) provides. The scheduler decides; it never touches a process, a socket or a card.
//!
//! | trait method | what it is on the box |
//! | --- | --- |
//! | `now`, `sleep` | the clock (a fake one in tests: sleeping advances it) |
//! | `snapshot` | the card sampler + the servers' states + the registry + pins and holds, one consistent view |
//! | `server` | one server's state read fresh (busy / holding), not from a snapshot |
//! | `start`, `stop` | start a model on a card in a server slot and wait until it answers; stop a slot's server |
//! | `free_video`, `free_chat` | let go of the video daemon's idle worker on a card; unload the idle chat model |
//! | `try_lock_kind` | the per-kind lock; None when another request holds it |
//! | `wait_loaded` | wait until a loading server answers |
//! | `note`, `cancelled` | a job's stage and a cancel flag |
//! | `learn_ask` | a server refused for want of memory: remember what it asked, say whether that is more than planned |

use crate::room::{Holder, Idleness, ModelInfo, Server, Snapshot};
use crate::Fail;

/// Whatever holds a lock until it is dropped (a mutex guard)
pub trait Guard {}
impl<T> Guard for T {}

pub trait Host: Idleness {
    /// Seconds on the host's clock
    fn now(&self) -> f64;
    /// Wait; a fake clock advances by it
    fn sleep(&self, secs: f64);
    /// The box now
    fn snapshot(&self) -> Snapshot;
    /// One server of a kind, read fresh
    fn server(&self, kind: &str, port: u16) -> Option<Server>;
    /// `id` started in server `slot` (listening on `port`) on card `gpu`, answering (or the error: a refusal for memory
    /// says "needs about X GiB on <card>; Y GiB are free")
    fn start(&self, kind: &str, id: &str, slot: usize, port: u16, gpu: usize) -> Result<(), Fail>;
    /// Stop the server in `slot`; `label` is what to call it in a log
    fn stop(&self, kind: &str, slot: usize, port: u16, label: &str) -> Result<(), Fail>;
    /// Let the video daemon's idle worker on `gpu` go, for model `for_id`
    fn free_video(&self, gpu: usize, for_id: &str) -> Result<(), Fail>;
    /// Unload the idle chat model `chat`
    fn free_chat(&self, chat: &str) -> Result<(), Fail>;
    /// The kind's lock, when it is free (the guard is dropped to release it)
    fn try_lock_kind<'a>(&'a self, kind: &str) -> Option<Box<dyn Guard + 'a>>;
    /// Wait until the loading server on `port` answers as `id`
    fn wait_loaded(&self, port: u16, id: &str) -> Result<(), Fail>;
    /// A job's stage and what it waits for
    fn note(&self, job: Option<&str>, stage: &str, why: &str);
    /// Whether a cancel was asked for the job
    fn cancelled(&self, job: &str) -> bool;
    /// A server refused to load for want of memory on `gpu`, planned for `want` MiB: remembered; whether it is more than planned
    fn learn_ask(&self, id: &str, err: &str, gpu: usize, want: u64) -> bool;

    // ---- nextsycl's additions; each default is what nextcuda does, so a host without them is unchanged ----

    /// The kind's lock, waited for: [`crate::place::ensure`] holds it around each try
    fn wait_lock_kind<'a>(&'a self, _kind: &str) -> Box<dyn Guard + 'a> {
        Box::new(())
    }
    /// A request was given the server on `port`, the kind's lock still held: count it now, or the next try sees that
    /// server with nothing to do
    fn placed(&self, _port: u16) {}
    /// Why nothing may load at all now (a GPU run holding the cards): what is loaded still answers, a request is
    /// refused with this, a job waits and its room wait starts over when it ends
    fn barred(&self) -> Option<String> {
        None
    }
    /// Why this model cannot load yet, whatever the room (its files moving): waited for like room
    fn not_ready(&self, _id: &str) -> Option<String> {
        None
    }
    /// The answer when the caller's own limit has passed, for a host that tracks it with the request
    /// (checked after each try and while a busy server drains)
    fn give_up(&self) -> Option<Fail> {
        None
    }
    /// What must go for the model's host memory (idle holders, in going-first order), or why it cannot load now
    fn ram_plan(&self, _snap: &Snapshot, _m: &ModelInfo) -> Result<Vec<Holder>, String> {
        Ok(Vec::new())
    }
    /// Wait until the host has the model's memory free after holders went; whether it came
    fn wait_ram(&self, _id: &str) -> bool {
        true
    }
    /// What card `g` has free now, for the wait after holders went (a host may have a cheaper way than a snapshot)
    fn free_mb(&self, g: usize) -> u64 {
        crate::room::card_free(&self.snapshot().cards, g)
    }
    /// Check again and take the room, atomically against every other kind's load, just before `start`
    /// ([`crate::room::may_start`] is the check). Err: not now, the next try plans again. What closes the gap in which
    /// two kinds plan for one card, or something starts beside an exclusive model still loading.
    fn reserve(&self, _m: &ModelInfo, _slot: usize, _port: u16, _gpu: usize, _want: u64) -> Result<(), String> {
        Ok(())
    }
    /// Among the cards with the room now, the first the model lists (the box's fastest card first) rather than the emptiest
    fn first_listed_card(&self) -> bool {
        false
    }
}
