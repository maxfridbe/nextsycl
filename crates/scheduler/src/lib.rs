//! The backplane's scheduler, as one self-contained crate (see README.md: the copy-and-adapt guide).
//!
//! Pure and deterministic. Everything the box does - read the cards, start and stop a model's server, sleep - is the
//! host's, through the traits in [`host`]; this crate decides.
//!
//! - [`profile`]: a model card's `concurrency` / `coexistence` (copies a card, exclusive / shared), validated;
//!   the bench's decision rules;
//! - [`rsvp`]: `rsvp_s` keep-alives (holds), the server cap, `max_willing_to_wait_s`, the `wait_too_long` error tag;
//! - [`room`]: the card / server / model snapshot types, room planning (`plan_room`, `pick_card`, exclusivity and
//!   `shared_with` as blockers), the `no_room` message;
//! - [`place`]: `place` - one try at a server for a model, over a [`host::Host`];
//! - [`shape`], [`history`], [`eta`]: request shapes per kind, the rolling history (storage behind a trait), ETAs,
//!   `would_accept`, the wait decisions;
//! - [`fake`]: a fake host on a fake clock, for the scenario suite and for testing a port.

pub mod eta;
pub mod fake;
pub mod history;
pub mod host;
pub mod place;
pub mod profile;
pub mod room;
pub mod rsvp;
pub mod shape;

/// The crate's version: bump it with a line in CHANGELOG.md when behaviour changes
pub const VERSION: &str = "0.1.0+nextsycl.1";

/// What a failed step answers: an HTTP-like status (kept for the host's error body) and the words
pub type Fail = (u16, String);
