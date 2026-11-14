//! nextsycl-serve: what turns an engine into a service - an OpenAI-compatible HTTP API (and a control socket) over a
//! kind's engine contract, the prompt cache, the GPUs' telemetry. It knows engines only through their kind's trait
//! (`nextsycl_llm::Engine`, `nextsycl_image::ImageEngine` in `image`, `nextsycl_audio::AudioEngine` in `audio`), never one
//! engine.

pub mod audio;
pub mod cache;
pub mod cancel;
pub mod describe;
pub mod carry;
pub mod drives;
pub mod files;
pub mod gpulock;
pub mod gpustat;
pub mod hold;
pub mod home;
pub mod idle;
pub mod maintenance;
pub mod model3d;
pub mod http;
pub mod image;
pub mod serve;
pub mod speechfmt;
pub mod switch;
pub mod telemetry;
pub mod video;
pub mod voices;
pub mod wfe;

pub use serve::Server;

/// An engine's error as its request's status: cancelled 499; the card out of memory (no room: retryable - a job waits
/// for the room, a caller may send it elsewhere) 503; else `other` (400: the request's fault)
pub(crate) fn engine_status(e: nextsycl_core::Error, other: u16) -> (u16, String) {
    (if e.0 == "cancelled" { 499 } else if e.is_no_room() { 503 } else { other }, e.0)
}

/// The release (yy.mmdd.###, from version.sh), reported by the API
pub const VERSION: &str = match option_env!("NS_VERSION") {
    Some(v) => v,
    None => "dev",
};

#[cfg(test)]
mod tests {
    use nextsycl_core::Error;

    #[test]
    fn out_of_card_memory_is_no_room() {
        let oom = "allocating 1207959552 bytes on Intel(R) Arc(TM) Pro B70 Graphics: device allocation of 1207959552 bytes failed";
        assert_eq!(super::engine_status(Error(oom.into()), 400).0, 503);
        assert_eq!(super::engine_status(Error("cancelled".into()), 400).0, 499);
        assert_eq!(super::engine_status(Error("size must be a multiple of 16".into()), 400).0, 400);
        assert_eq!(super::engine_status(Error("no mask".into()), 500).0, 500);
    }
}
