//! The two kernels of the image library this engine takes beyond the shared ones (nsd): the upscaler's activation
//! (`ns_image_up_act`, kernels/image/upscale/up.h: its sigmoid) and element-wise op (`ns_image_up_add`: a scale, a
//! product) - QuickGELU and Swish from them. Bound by name, as the upscaler binds them (an engine does not depend on
//! another engine's crate).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    /// (gpu, x half [N, C] in place, N, C, kind (4: sigmoid), a, slopes, n_slopes)
    pub act: unsafe extern "C" fn(G, M, i64, i64, c_int, f32, *const f32, i64) -> c_int,
    /// (gpu, a, b, n halves, op (0: ca a + cb b, 1: a b), ca, cb, out)
    pub add: unsafe extern "C" fn(G, P, P, i64, c_int, f32, f32, M) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api() -> nextsycl_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        let act = lib.symbol("ns_image_up_act");
        let add = lib.symbol("ns_image_up_add");
        if act.is_null() || add.is_null() {
            return Err("ns_image_up_act / ns_image_up_add are not in the kernel library (the image library, rebuilt: kernels/build.sh)".into());
        }
        // SAFETY: declared in up.h with these signatures.
        unsafe {
            Ok(Api {
                act: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(G, M, i64, i64, c_int, f32, *const f32, i64) -> c_int>(act),
                add: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(G, P, P, i64, c_int, f32, f32, M) -> c_int>(add),
            })
        }
    })
    .as_ref()
    .map_err(|e| nextsycl_core::Error(e.clone()))
}
