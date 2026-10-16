//! RigNet (Xu et al., SIGGRAPH 2020; github.com/zhan-xu/RigNet): a skeleton and skin weights for a mesh. GPLv3, as
//! RigNet's code: this crate is a port of it and is kept apart from the MIT ones.
//!
//! Not ported yet: `load` says so.

pub mod check;

use std::sync::Arc;

use nextsycl_core::Gpu;
use nextsycl_model3d::{Error, LoadOptions, Model3dEngine, Model3dKind, ModelFiles, Result};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "rignet";

/// The files it needs, by role
pub const ROLES: &[&str] = &["models"];

pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "RigNet (a skeleton and skin weights: GMEdgeNet, RootNet, BoneNet, SkinNet; GPLv3)", roles: ROLES, load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(RigNet::load(files, gpu, o, log)?))
}

pub struct RigNet {
    load_s: f64,
}

impl RigNet {
    pub fn load(_files: &ModelFiles, _gpu: &Arc<Gpu>, _o: &LoadOptions, _log: &mut dyn FnMut(String)) -> Result<RigNet> {
        Err(Error(format!("{ARCH}: not ported yet")))
    }
}

impl Model3dEngine for RigNet {
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
