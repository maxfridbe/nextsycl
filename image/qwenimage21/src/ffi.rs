//! This engine's own kernels (`kernels/image/qwenimage21/qi21.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub modulate: unsafe extern "C" fn(G, M, i64, i64, *const f32) -> c_int,
    pub gelu_tanh: unsafe extern "C" fn(G, M, i64) -> c_int,
    pub silu: unsafe extern "C" fn(G, M, i64) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    pub dupup_add: unsafe extern "C" fn(G, P, i64, i64, i64, i64, c_int, M) -> c_int,
    pub to_rgba8: unsafe extern "C" fn(G, P, i64, i64, *mut u8) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api() -> nextsycl_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (the image library, rebuilt: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in qi21.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            modulate: sym!("ns_image_qi21_modulate"),
            gelu_tanh: sym!("ns_image_qi21_gelu_tanh"),
            silu: sym!("ns_image_qi21_silu"),
            axpy: sym!("ns_image_qi21_axpy"),
            to_half: sym!("ns_image_qi21_to_half"),
            to_float: sym!("ns_image_qi21_to_float"),
            up2: sym!("ns_image_qi21_up2"),
            dupup_add: sym!("ns_image_qi21_dupup_add"),
            to_rgba8: sym!("ns_image_qi21_to_rgba8"),
        })
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
