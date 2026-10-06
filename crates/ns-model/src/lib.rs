//! The model architectures the runtime knows, each as: its geometry (read from a file's metadata), its layer
//! kinds, and every tensor by ROLE - the runtime's own names - resolved against the file's naming scheme and
//! checked for shape at load. Files of one architecture come from different converters with different names
//! (llama.cpp's PR names and antirez ds4's, for GLM5-Next); the roles are what the rest of the runtime uses.

pub mod glm5next;

use ns_gguf::Gguf;

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<ns_gguf::Error> for Error {
    fn from(e: ns_gguf::Error) -> Error {
        Error(e.0)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Which architecture a file holds, by `general.architecture`.
pub fn architecture(g: &Gguf) -> Result<&'static str> {
    match g.architecture() {
        "glm5-next" | "glm5next" => Ok("glm5-next"),
        other => Err(Error(format!("architecture {other:?} is not one this runtime knows (glm5-next; qwen4exp to come)"))),
    }
}
