//! UniRig (VAST-AI, SIGGRAPH 2025; github.com/VAST-AI-Research/UniRig, MIT): a skeleton and skin weights for a mesh.
//!
//! Not ported yet: `load` says so.

pub mod check;

use std::sync::Arc;

use nextsycl_core::Gpu;
use nextsycl_model3d::{Error, LoadOptions, Model3dEngine, Model3dKind, ModelFiles, Result};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "unirig";

/// The files it needs, by role
pub const ROLES: &[&str] = &["skeleton", "skin"];

pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "UniRig (a skeleton from an autoregressive decoder, skin weights from Point Transformer V3)", roles: ROLES, load,
                  options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(UniRig::load(files, gpu, o, log)?))
}

pub struct UniRig {
    load_s: f64,
}

impl UniRig {
    pub fn load(_files: &ModelFiles, _gpu: &Arc<Gpu>, _o: &LoadOptions, _log: &mut dyn FnMut(String)) -> Result<UniRig> {
        Err(Error(format!("{ARCH}: not ported yet")))
    }
}

impl Model3dEngine for UniRig {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn abilities(&self) -> nextsycl_model3d::Abilities {
        nextsycl_model3d::Abilities { rig: true, ..Default::default() }
    }
    fn defaults(&self) -> nextsycl_model3d::Defaults {
        nextsycl_model3d::Defaults::default()
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
}
