//! The kernels the removers call: their own (`kernels/image/rembg/rb.h`) and the shared vision ones
//! (`kernels/diffusion/vision.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{Error, Result};

type G = *mut c_void;

pub struct Api {
    pub maxpool2: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32) -> c_int,
    pub resize: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, i64, *mut f32, i64) -> c_int,
    pub bias_act: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32, c_int) -> c_int,
    pub sigmoid: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
}

pub struct Vision {
    #[allow(clippy::type_complexity)]
    pub im2col: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub copy_cols: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}
unsafe impl Send for Vision {}
unsafe impl Sync for Vision {}

macro_rules! sym {
    ($lib:expr, $name:literal) => {{
        let p = $lib.symbol($name);
        if p.is_null() {
            return Err(format!("{} is not in the kernel library (the image library, rebuilt: kernels/build.sh)", $name));
        }
        // SAFETY: the symbol is declared in rb.h / vision.h with the field's signature.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            std::mem::transmute::<*mut c_void, _>(p)
        }
    }};
}

pub fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        Ok(Api {
            maxpool2: sym!(lib, "ns_image_rb_maxpool2"),
            resize: sym!(lib, "ns_image_rb_resize"),
            bias_act: sym!(lib, "ns_image_rb_bias_act"),
            sigmoid: sym!(lib, "ns_image_rb_sigmoid"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

pub fn vision() -> Result<&'static Vision> {
    static V: OnceLock<std::result::Result<Vision, String>> = OnceLock::new();
    V.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        Ok(Vision { im2col: sym!(lib, "ns_vision_im2col"), copy_cols: sym!(lib, "ns_vision_copy_cols"), add: sym!(lib, "ns_vision_add") })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("{what}: {why}")))
}
