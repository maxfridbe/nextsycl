//! The forward passes. `glm5next`: GLM-5.3-Flash, as `docs/glm5next.md` writes it down.

pub mod glm5next;

pub use ns_core::{Error, Result};

/// Called with a tensor's name (llama.cpp's graph names: `attn_norm-3`, `l_out-44`, ...) and its float32 values on
/// the GPU, at each point a reference dump can be compared with.
pub type Tap<'a> = &'a mut dyn FnMut(&str, &ns_core::DevBuf) -> Result<()>;
