//! This engine's own kernels (`kernels/model3d/hunyuan3d/h3d.h`), bound from the 3D kernel library by name. The
//! shared diffusion kernels (`nextsycl_diffusion::kernels`) do the rest.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub ln: unsafe extern "C" fn(G, P, c_int, i64, i64, f32, *const f32, *const f32, *const f32, *const f32, M, c_int) -> c_int,
    pub head_norm: unsafe extern "C" fn(G, M, i64, i64, i64, i64, *const f32, *const f32, f32, c_int) -> c_int,
    pub silu: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, P, c_int, i64, i64, *const f32) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    pub cfg: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, i64, f32) -> c_int,
    pub fourier: unsafe extern "C" fn(G, *const f32, *const i32, i64, i64, i64, f32, f32, c_int, c_int, M) -> c_int,
    pub route: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut i32, *mut f32) -> c_int,
    pub gather: unsafe extern "C" fn(G, P, *const i32, i64, i64, M) -> c_int,
    pub scatter_add: unsafe extern "C" fn(G, P, *const i32, *const f32, i64, i64, *mut f32) -> c_int,
    pub group_norm: unsafe extern "C" fn(G, P, i64, i64, i64, i64, *const f32, *const f32, f32, c_int, *mut f32, M) -> c_int,
    pub geglu: unsafe extern "C" fn(G, P, i64, i64, M) -> c_int,
    pub pad2d: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, i64, i64, M) -> c_int,
    pub concat: unsafe extern "C" fn(G, P, i64, P, i64, i64, M) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, i64, M) -> c_int,
    pub add_half: unsafe extern "C" fn(G, M, P, i64, f32) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// The table, bound once (an error when the library lacks a symbol: an older build)
pub fn api() -> nextsycl_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (the 3D library, rebuilt: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in h3d.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            ln: sym!("ns_model3d_hunyuan3d_ln"),
            head_norm: sym!("ns_model3d_hunyuan3d_head_norm"),
            silu: sym!("ns_model3d_hunyuan3d_silu"),
            to_half: sym!("ns_model3d_hunyuan3d_to_half"),
            to_float: sym!("ns_model3d_hunyuan3d_to_float"),
            add: sym!("ns_model3d_hunyuan3d_add"),
            axpy: sym!("ns_model3d_hunyuan3d_axpy"),
            cfg: sym!("ns_model3d_hunyuan3d_cfg"),
            fourier: sym!("ns_model3d_hunyuan3d_fourier"),
            route: sym!("ns_model3d_hunyuan3d_route"),
            gather: sym!("ns_model3d_hunyuan3d_gather"),
            scatter_add: sym!("ns_model3d_hunyuan3d_scatter_add"),
            group_norm: sym!("ns_model3d_hunyuan3d_group_norm"),
            geglu: sym!("ns_model3d_hunyuan3d_geglu"),
            pad2d: sym!("ns_model3d_hunyuan3d_pad2d"),
            concat: sym!("ns_model3d_hunyuan3d_concat"),
            up2: sym!("ns_model3d_hunyuan3d_up2"),
            add_half: sym!("ns_model3d_hunyuan3d_add_half"),
        })
    })
    .as_ref()
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The silo's kernels (kernels/model3d/hunyuan3d/silo/silo.h, dist/silo/libnextsycl-hunyuan3d.so)
#[allow(clippy::type_complexity)]
pub struct Silo {
    pub path: std::path::PathBuf,
    pub group_norm_supported: unsafe extern "C" fn(i64, i64, i64, i64) -> c_int,
    pub group_norm_scratch: unsafe extern "C" fn(i64, i64, i64, i64) -> i64,
    pub group_norm: unsafe extern "C" fn(G, P, i64, i64, i64, i64, *const f32, *const f32, f32, c_int, *mut f32, M) -> c_int,
    pub ln_supported: unsafe extern "C" fn(i64) -> c_int,
    pub ln: unsafe extern "C" fn(G, P, c_int, i64, i64, f32, *const f32, *const f32, *const f32, *const f32, M, c_int) -> c_int,
    pub geglu_supported: unsafe extern "C" fn(i64, i64) -> c_int,
    pub geglu: unsafe extern "C" fn(G, P, i64, i64, M) -> c_int,
    pub add_supported: unsafe extern "C" fn(i64, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, P, c_int, i64, i64, *const f32) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo (None: the engine's own kernels throughout - no file, or NS_SILO=0)
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static S: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo("hunyuan3d").map_err(|e| e.0)? else { return Ok(None) };
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
            path: lib.path.clone(),
            group_norm_supported: sym!("ns_model3d_hy_silo_group_norm_supported"),
            group_norm_scratch: sym!("ns_model3d_hy_silo_group_norm_scratch"),
            group_norm: sym!("ns_model3d_hy_silo_group_norm"),
            ln_supported: sym!("ns_model3d_hy_silo_ln_supported"),
            ln: sym!("ns_model3d_hy_silo_ln"),
            geglu_supported: sym!("ns_model3d_hy_silo_geglu_supported"),
            geglu: sym!("ns_model3d_hy_silo_geglu"),
            add_supported: sym!("ns_model3d_hy_silo_add_supported"),
            add: sym!("ns_model3d_hy_silo_add"),
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
