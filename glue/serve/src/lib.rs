//! nextsycl-serve: what turns an engine into a service - an OpenAI-compatible HTTP API (and a control socket) over a
//! kind's engine contract, the prompt cache, the GPUs' telemetry. It knows engines only through their kind's trait
//! (`nextsycl_llm::Engine`), never one engine.

pub mod cache;
pub mod http;
pub mod serve;
pub mod telemetry;

pub use serve::Server;

/// The release (yy.mmdd.###, from version.sh), reported by the API
pub const VERSION: &str = match option_env!("NS_VERSION") {
    Some(v) => v,
    None => "dev",
};
