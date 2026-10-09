//! nextsycl-serve: what turns an engine into a service - an OpenAI-compatible HTTP API (and a control socket) over a
//! kind's engine contract, the prompt cache, the GPUs' telemetry. It knows engines only through their kind's trait
//! (`nextsycl_llm::Engine`, `nextsycl_image::ImageEngine` in `image`, `nextsycl_audio::AudioEngine` in `audio`), never one
//! engine.

pub mod audio;
pub mod cache;
pub mod gpustat;
pub mod http;
pub mod image;
pub mod serve;
pub mod switch;
pub mod telemetry;
pub mod video;

pub use serve::Server;

/// The release (yy.mmdd.###, from version.sh), reported by the API
pub const VERSION: &str = match option_env!("NS_VERSION") {
    Some(v) => v,
    None => "dev",
};
