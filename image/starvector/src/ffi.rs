//! The kernels of the image library this engine takes beyond the shared ones (nsd): the upscaler's activation
//! (`ns_image_up_act`, kernels/image/upscale/up.h: its sigmoid) and element-wise op (`ns_image_up_add`: a scale, a
//! product) - QuickGELU and Swish from them. Bound by name, as the upscaler binds them (an engine does not depend on
//! another engine's crate). And the engine's own silo (kernels/image/starvector/silo): the decode attention.

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

/// The silo's name (`kernels/image/starvector/silo` -> `dist/silo/libnextsycl-starvector.so`)
pub const SILO: &str = "starvector";

/// The silo's decode attention (silo.h)
pub struct Silo {
    /// (gpu, beams, keys, heads, head size): 1 when the silo runs it
    pub attn_supported: unsafe extern "C" fn(G, i64, i64, i64, i64) -> c_int,
    /// (gpu, q half [beams, heads x head], each beam's k and v caches half [keys, head], beams, keys, heads, head,
    /// out half [beams, heads x head])
    pub decode_attn: unsafe extern "C" fn(G, P, *const P, *const P, i64, i64, i64, i64, M) -> c_int,
    /// (gpu, rows, row length): 1 when the silo's layer norm runs it
    pub ln_supported: unsafe extern "C" fn(G, i64, i64) -> c_int,
    /// (gpu, x float32 [rows, c], rows, c, weight, bias float32 [c], eps, out half [rows, c])
    pub layer_norm: unsafe extern "C" fn(G, P, i64, i64, P, P, f32, M) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo, when built (None: NS_SILO=0, or no file - the shared attention throughout)
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
            attn_supported: sym!("ns_image_starvector_silo_attn_supported"),
            decode_attn: sym!("ns_image_starvector_silo_decode_attn"),
            ln_supported: sym!("ns_image_starvector_silo_ln_supported"),
            layer_norm: sym!("ns_image_starvector_silo_layer_norm"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| nextsycl_core::Error(e.clone()))
}
