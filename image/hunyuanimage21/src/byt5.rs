//! The glyph encoder: Glyph-SDXL-v2's byT5-small (crates/hunyuan-text, shared with HunyuanVideo-1.5) with HunyuanImage
//! 2.1's rules: the pieces in "...", ‘...’ and “...” (in that order, by kind), at most 128 tokens.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::Result;
use nextsycl_diffusion::kernels::Nsd;

pub use nextsycl_hunyuan_text::byt5::{bucket, Byt5, DIM};
use nextsycl_hunyuan_text::byt5::Quotes;

use crate::ffi::K;

pub const MAX_TOKENS: usize = 128;

/// The quoted pieces of a prompt as the pipeline formats them, or None when it quotes nothing
pub fn quoted(prompt: &str) -> Option<String> {
    nextsycl_hunyuan_text::byt5::quoted(prompt, Quotes::ByKind)
}

/// ByT5's ids, at most 128
pub fn ids(text: &str) -> Vec<u32> {
    nextsycl_hunyuan_text::byt5::ids(text, MAX_TOKENS)
}

pub fn load(file: &Path, nsd: &Nsd, k: &K, log: &mut dyn FnMut(String)) -> Result<Byt5> {
    Byt5::load(file, nsd, Arc::new(k.clone()), log)
}
