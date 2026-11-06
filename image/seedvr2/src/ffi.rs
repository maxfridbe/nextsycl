//! This engine's own kernels (`kernels/image/seedvr2/svr.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    // the VAE's (conv.cpp, svr.cpp)
    pub conv3d: unsafe extern "C" fn(G, P, i64, i64, i64, i64, P, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, *const f32, M) -> c_int,
    pub group_norm: unsafe extern "C" fn(G, P, i64, i64, i64, i64, *const f32, *const f32, f32, c_int, M) -> c_int,
    pub shuffle_up: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, i64, c_int, M) -> c_int,
    pub softmax_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M) -> c_int,
    pub add: unsafe extern "C" fn(G, M, P, i64) -> c_int,
    // the frames'
    pub resize: unsafe extern "C" fn(G, *const u8, i64, i64, i64, i64, i64, i64, i64, i64, i64, M) -> c_int,
    pub colorfix: unsafe extern "C" fn(G, P, P, i64, i64, i64, *mut f32, M) -> c_int,
    pub to_rgb8: unsafe extern "C" fn(G, P, i64, *mut u8) -> c_int,
    // the DiT's
    pub to_bf16: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_f32: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    pub norm_mod: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, f32, *const f32, *const f32, M) -> c_int,
    pub gate_add: unsafe extern "C" fn(G, *mut f32, P, i64, i64, *const f32) -> c_int,
    pub qk_norm_rope: unsafe extern "C" fn(G, M, i64, i64, i64, i64, *const f32, *const f32, f32, *const f32, i64) -> c_int,
    pub gather: unsafe extern "C" fn(G, P, i64, i64, P, i64, *const i32, i64, i64, M) -> c_int,
    pub text_mean: unsafe extern "C" fn(G, P, *const i32, i64, i64, i64, M) -> c_int,
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
                // SAFETY: the symbol is declared in svr.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            conv3d: sym!("ns_image_seedvr2_conv3d"),
            group_norm: sym!("ns_image_seedvr2_group_norm"),
            shuffle_up: sym!("ns_image_seedvr2_shuffle_up"),
            softmax_rows: sym!("ns_image_seedvr2_softmax_rows"),
            add: sym!("ns_image_seedvr2_add"),
            resize: sym!("ns_image_seedvr2_resize"),
            colorfix: sym!("ns_image_seedvr2_colorfix"),
            to_rgb8: sym!("ns_image_seedvr2_to_rgb8"),
            to_bf16: sym!("ns_image_seedvr2_to_bf16"),
            to_f32: sym!("ns_image_seedvr2_to_f32"),
            norm_mod: sym!("ns_image_seedvr2_norm_mod"),
            gate_add: sym!("ns_image_seedvr2_gate_add"),
            qk_norm_rope: sym!("ns_image_seedvr2_qk_norm_rope"),
            gather: sym!("ns_image_seedvr2_gather"),
            text_mean: sym!("ns_image_seedvr2_text_mean"),
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

/// The silo's name (`kernels/image/seedvr2/silo` -> `dist/silo/libnextsycl-seedvr2.so`)
pub const SILO: &str = "seedvr2";

/// The silo's kernels (silo.h)
pub struct Silo {
    pub supported: unsafe extern "C" fn(G) -> c_int,
    pub conv_taps: unsafe extern "C" fn(G, *const P, i64, i64, i64, i64, P, i64, i64, i64, i64, i64, i64, *const f32, M) -> c_int,
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
        Ok(Some(Silo { supported: sym!("ns_image_seedvr2_silo_supported"), conv_taps: sym!("ns_image_seedvr2_silo_conv_taps") }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| nextsycl_core::Error(e.clone()))
}
