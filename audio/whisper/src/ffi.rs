//! The one kernel of the audio library this engine needs beyond the shared diffusion ones (nsd: products, norms,
//! GELU, attention, the 1-D convolution): the MiniMax Music engine's float32 transpose (`kernels/audio/minimaxmusic3/
//! mm3.h`), which turns the convolutions' channels-first output into the transformer's rows.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;

pub struct Api {
    pub transpose: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api() -> nextsycl_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        let p = lib.symbol("ns_audio_mm3_transpose");
        if p.is_null() {
            return Err("ns_audio_mm3_transpose is not in the kernel library (the audio library, rebuilt: ./build.sh kernels)".into());
        }
        // SAFETY: declared in mm3.h with this signature.
        Ok(Api { transpose: unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int>(p) } })
    })
    .as_ref()
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextsycl_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextsycl_core::Error(format!("{what}: {why}")))
}
