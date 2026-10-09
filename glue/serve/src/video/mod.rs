//! The video service: the daemon (`daemon`: the job queue, a worker process a GPU, the card lock and the front
//! end's model switch - H3's `h3d daemon`), engine-agnostic.

pub mod daemon;
pub mod signals;
