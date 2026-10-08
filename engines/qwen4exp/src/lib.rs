//! Qwen3.8-Flash-Next (`qwen4exp`): its model description (`model`: the geometry from the file's metadata, every
//! tensor by role) and its engine, behind the runtime's `Engine` contract (`ns-runtime`). `kind()` is its registry
//! entry. Its kernels are Strata's SYCL port (`kernels/strata`, the user's intel-arc branch), wrapped in
//! `kernels/engines/qwen4exp/`.

pub mod model;
pub mod tools;

use std::sync::Arc;

use ns_gguf::Gguf;
use ns_runtime::{Engine, EngineKind, LoadOptions, Result};

/// The model description's errors (a file's layout against what the engine needs)
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.into())
    }
}

pub type ModelResult<T> = std::result::Result<T, Error>;

/// This engine's registry entry
pub fn kind() -> EngineKind {
    EngineKind { archs: &["qwen4exp"], name: "Qwen3.8-Flash-Next (Gated DeltaNet + QSA, hyper-connections, PLE, 512 experts)", load, info: tools::info,
                 kernels: None }
}

fn load<'g>(f: &'g Gguf, _gpus: &[Arc<ns_core::Gpu>], _o: &LoadOptions, _log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    let m = model::Model::open(f).map_err(|e| ns_core::Error(e.0))?;
    let errs = m.check();
    if !errs.is_empty() {
        return Err(ns_core::Error(format!("{}: {}", f.paths[0].display(), errs.join("; "))));
    }
    Err(ns_core::Error("qwen4exp: the model description is in place, the engine is not yet (nextsycl info shows the file)".into()))
}
