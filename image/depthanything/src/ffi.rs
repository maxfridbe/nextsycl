//! The kernels the depth network calls: its own (`kernels/image/depthanything/da.h`) and the shared vision ones
//! (`kernels/diffusion/vision.h`: im2col, the align-corners resize, add), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{Error, Result};

type G = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub embed: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, *const f32, i64, i64, *mut f32) -> c_int,
    pub layernorm: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, f32, *mut f32) -> c_int,
    pub bias_act: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32, c_int) -> c_int,
    pub act: unsafe extern "C" fn(G, *const f32, i64, c_int, *mut f32) -> c_int,
    pub ls_add: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, *const f32, i64, i64) -> c_int,
    pub attention: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32) -> c_int,
    pub shuffle: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, *const f32, *mut f32) -> c_int,
    pub resize: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, i64, *mut f32, i64) -> c_int,
}

#[allow(clippy::type_complexity)]
pub struct Vision {
    pub im2col: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub resize_ac: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, i64, *mut f32, i64) -> c_int,
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
        // SAFETY: the symbol is declared in da.h / vision.h with the field's signature.
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
            embed: sym!(lib, "ns_image_da_embed"),
            layernorm: sym!(lib, "ns_image_da_layernorm"),
            bias_act: sym!(lib, "ns_image_da_bias_act"),
            act: sym!(lib, "ns_image_da_act"),
            ls_add: sym!(lib, "ns_image_da_ls_add"),
            attention: sym!(lib, "ns_image_da_attention"),
            shuffle: sym!(lib, "ns_image_da_shuffle"),
            resize: sym!(lib, "ns_image_da_resize"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

pub fn vision() -> Result<&'static Vision> {
    static V: OnceLock<std::result::Result<Vision, String>> = OnceLock::new();
    V.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        Ok(Vision { im2col: sym!(lib, "ns_vision_im2col"), resize_ac: sym!(lib, "ns_vision_resize_ac"), add: sym!(lib, "ns_vision_add") })
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

/// The silo's name (`kernels/image/depthanything/silo` -> `dist/silo/libnextsycl-depthanything.so`)
pub const SILO: &str = "depthanything";

/// The silo's attention (silo.h)
pub struct Silo {
    pub supported: unsafe extern "C" fn(G, i64) -> c_int,
    pub attention: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32) -> c_int,
    /// the attention as products (gemm_attn.cpp): heads split for them, a row's softmax, the heads merged
    pub split: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32, *mut f32, *mut f32) -> c_int,
    pub softmax: unsafe extern "C" fn(G, *mut f32, i64, i64, f32) -> c_int,
    pub merge: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo, when built (None: NS_SILO=0, or no file - the engine's own kernels throughout)
pub fn silo() -> Result<Option<&'static Silo>> {
    static S: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
        Ok(Some(Silo { supported: sym!(lib, "ns_image_da_silo_supported"), attention: sym!(lib, "ns_image_da_silo_attention"),
                       split: sym!(lib, "ns_image_da_silo_split"), softmax: sym!(lib, "ns_image_da_silo_softmax"), merge: sym!(lib, "ns_image_da_silo_merge") }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| Error(e.clone()))
}
