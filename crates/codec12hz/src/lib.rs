//! The Qwen3-TTS 12 Hz speech codec (Qwen's Qwen3-TTS-Tokenizer-12Hz, the `speech_tokenizer/` of the Qwen3-TTS
//! checkpoints; Breeze TTS 2 ships the same file byte for byte as its `audio_tokenizer/`): the decoder (decoder.rs, a
//! frame's 16 codes to 1,920 samples at 24 kHz) and the encoder (encoder.rs, a recording to its frames' codes), with
//! the resampler a recording goes through first (resample.rs).
//!
//! Its kernels are in the audio kernel library - the codec's own (`kernels/audio/qwen3tts/q3t.h`: SnakeBeta, the
//! depthwise convolution, windowed attention, RoPE, layer-scaled residuals, ELU) and MiniMax Music's convolutions on
//! oneDNN (`kernels/audio/minimaxmusic3/mm3.h`), bound here by name - so a process that uses it opens that library
//! (`nextsycl_core::use_kind("audio")`); the linear layers and norms are the shared diffusion kernels' (`Nsd`). Float32
//! throughout, the checkpoint's own precision.

pub mod decoder;
pub mod encoder;
mod ffi;
pub mod ops;
pub mod resample;

pub use decoder::{Codec, HOP, RATE};
pub use encoder::Encoder;
pub use ops::{Ops, Weights};
pub use resample::resample;

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
