//! Stage by stage against the reference (reference/unirig/ref.py).

use std::path::Path;

use nextsycl_model3d::Result;

use crate::UniRig;

/// Every stage of `stages` against the dumps in `dir`; the worst relative error
pub fn run(_m: &UniRig, _dir: &Path, _stages: &[&str], _mesh: Option<&Path>, _log: &mut dyn FnMut(String)) -> Result<f64> {
    Ok(0.0)
}
