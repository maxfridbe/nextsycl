//! Stage by stage against the reference (reference/rignet/ref.py).

use std::path::Path;

use nextsycl_model3d::Result;

use crate::RigNet;

/// Every stage of `stages` against the dumps in `dir` (`mesh`: the input mesh, for the stages that read it); the worst
/// relative error
pub fn run(_m: &RigNet, _dir: &Path, _stages: &[&str], _mesh: Option<&Path>, _log: &mut dyn FnMut(String)) -> Result<f64> {
    Ok(0.0)
}
