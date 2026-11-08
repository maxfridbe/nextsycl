//! This engine's own kernels (`kernels/image/zimage/zi.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub ln_mod: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const f32, M) -> c_int,
    pub rms_mod: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, f32, *const f32, M) -> c_int,
    pub gate_add: unsafe extern "C" fn(G, *mut f32, i64, i64, P, *const f32) -> c_int,
    pub to_half_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M, i64) -> c_int,
    pub swiglu_scaled: unsafe extern "C" fn(G, P, i64, i64, f32, M) -> c_int,
    pub swiglu_f: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    // the VAE's
    pub group_norm: unsafe extern "C" fn(G, P, i64, i64, i64, f32, *const f32, *const f32, c_int, M, *mut f32) -> c_int,
    pub softmax_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    pub to_rgb8: unsafe extern "C" fn(G, P, i64, i64, *mut u8) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the image library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in zi.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            ln_mod: sym!("ns_image_zi_ln_mod"),
            rms_mod: sym!("ns_image_zi_rms_mod"),
            gate_add: sym!("ns_image_zi_gate_add"),
            to_half_rows: sym!("ns_image_zi_to_half_rows"),
            swiglu_scaled: sym!("ns_image_zi_swiglu_scaled"),
            swiglu_f: sym!("ns_image_zi_swiglu_f"),
            axpy: sym!("ns_image_zi_axpy"),
            to_half: sym!("ns_image_zi_to_half"),
            to_float: sym!("ns_image_zi_to_float"),
            group_norm: sym!("ns_image_zi_group_norm"),
            softmax_rows: sym!("ns_image_zi_softmax_rows"),
            up2: sym!("ns_image_zi_up2"),
            to_rgb8: sym!("ns_image_zi_to_rgb8"),
        })
    })
    .as_ref()
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The scale the gated products are made at in half (they pass half's range: 6.5e4 in the DiT's 15th block at the
/// README's prompt; the next linear's float32 output multiplied back, or its norm's epsilon scaled the same)
pub const ACT_SCALE: f32 = 1.0 / 256.0;
/// The scale the DiT's q | k | v are made half at (|k| 5.6e3, |v| 1.9e3 at the README's prompt: 1/16 keeps them in range
/// without pushing their small values into half's subnormals as 1/256 does)
pub const QKV_SCALE: f32 = 1.0 / 16.0;

/// The silo's name (`kernels/image/zimage/silo` -> `dist/silo/libnextsycl-zimage.so`)
pub const SILO: &str = "zimage";

/// The silo's fused passes (silo.h)
pub struct Silo {
    pub supported: unsafe extern "C" fn(G, i64, i64) -> c_int,
    pub norm_gate_norm: unsafe extern "C" fn(G, *const f32, *mut f32, i64, i64, *const f32, f32, *const f32, *const f32, f32, *const f32, M) -> c_int,
    pub qk_rope: unsafe extern "C" fn(G, M, i64, i64, i64, i64, i64, *const f32, *const f32, f32, *const f32) -> c_int,
    pub swiglu: unsafe extern "C" fn(G, P, i64, i64, f32, M) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo, when built (None: NS_SILO=0, or no file - the engine's own kernels throughout)
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static S: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{}: no {} (rebuild it: ./build.sh kernels)", lib.path.display(), $name));
                }
                // SAFETY: the symbol is declared in silo.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Some(Silo {
            supported: sym!("ns_image_zimage_silo_supported"),
            norm_gate_norm: sym!("ns_image_zimage_silo_norm_gate_norm"),
            qk_rope: sym!("ns_image_zimage_silo_qk_rope"),
            swiglu: sym!("ns_image_zimage_silo_swiglu"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
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
