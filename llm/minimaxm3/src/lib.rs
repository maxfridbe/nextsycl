//! MiniMax-M3.1-Flash (`minimax-m3`): its model description (`model`: the geometry from the metadata, every tensor by
//! role). The engine and its silo (`kernels/llm/minimaxm3/silo`) are not written yet: docs/ports/minimax-m31.md.

/// The architecture this engine is, and the silo's name
pub const ARCH: &str = "minimaxm3";

pub mod model;

/// The model description's errors (a file's layout against what the engine needs)
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

pub type ModelResult<T> = std::result::Result<T, Error>;
