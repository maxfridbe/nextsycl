//! The text encoder: Qwen2.5-VL-7B's hidden_states[-3] (crates/hunyuan-text, shared with HunyuanVideo-1.5) behind
//! HunyuanImage 2.1's template: the official PROMPT_TEMPLATE_ENCODE_V2, the first 34 tokens - the template's system
//! turn - dropped from the output (crop_start), at most 1,000 tokens kept after them.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::Result;
use nextsycl_diffusion::kernels::Nsd;

pub use nextsycl_hunyuan_text::te::{TextEncoder, LAYERS_USED};

use crate::ffi::K;

pub const TEMPLATE_HEAD: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
pub const TEMPLATE_TAIL: &str = "<|im_end|>";
/// the template's tokens dropped from the output
pub const CROP: usize = 34;
/// the most tokens the pipeline encodes (its text_len 1000 after the crop)
pub const MAX_TOKENS: usize = 1000 + CROP;

/// HunyuanImage 2.1's template
pub fn template() -> nextsycl_hunyuan_text::te::Template {
    nextsycl_hunyuan_text::te::Template { head: TEMPLATE_HEAD.into(), tail: TEMPLATE_TAIL.into(), crop: CROP, max_tokens: MAX_TOKENS, empty_as_space: false }
}

pub fn load(file: &Path, tokenizer: &Path, nsd: &Nsd, k: &K, log: &mut dyn FnMut(String)) -> Result<TextEncoder> {
    TextEncoder::load(file, tokenizer, nsd, Arc::new(k.clone()), template(), log)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_template_crop_is_its_system_turn() {
        // the template's head is 34 tokens of Qwen's vocabulary up to and including "user\n" (checked against the
        // pipeline's tokenizer in the reference dumps); the prompt starts right after
        assert!(super::TEMPLATE_HEAD.ends_with("<|im_start|>user\n"));
        assert_eq!(super::CROP, 34);
        assert_eq!(super::MAX_TOKENS, 1034);
    }
}
