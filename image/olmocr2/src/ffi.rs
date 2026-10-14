//! This engine's own C ABI (`kernels/image/olmocr2/olm.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type F = *const f32;
type FM = *mut f32;

pub struct Api {
    pub to_half: unsafe extern "C" fn(G, F, *mut c_void, i64) -> c_int,
    /// (src, src stride, dst, dst stride, rows, cols)
    pub copy_rows: unsafe extern "C" fn(G, F, i64, FM, i64, i64, i64) -> c_int,
    /// (src, C, idx, n, out)
    pub gather: unsafe extern "C" fn(G, F, i64, *const i32, i64, FM) -> c_int,
    /// (q, k, v, stride, N, H, D, lo, hi, out)
    pub seg_attn: unsafe extern "C" fn(G, F, F, F, i64, i64, i64, i64, *const i32, *const i32, FM) -> c_int,
    /// (qs, d, x, n, k, bias, y)
    pub q8r_gemv: unsafe extern "C" fn(G, *const i8, *const c_void, F, i64, i64, F, FM) -> c_int,
    /// (qs, d, n, k, half out)
    pub q8r_half: unsafe extern "C" fn(G, *const i8, *const c_void, i64, i64, *mut c_void) -> c_int,
    /// (n, Hq, D): the scratch attn1 needs, in floats
    pub attn1_scratch: unsafe extern "C" fn(i64, i64, i64) -> i64,
    /// (q, k, v, n, kv stride, Hq, Hkv, D, scratch, out)
    pub attn1: unsafe extern "C" fn(G, F, F, F, i64, i64, i64, i64, i64, FM, FM) -> c_int,
}

/// The table, bound once (an error when the library lacks a symbol: an older build)
pub fn api() -> nextsycl_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (rebuild it: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in olm.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            to_half: sym!("ns_image_olm_to_half"),
            copy_rows: sym!("ns_image_olm_copy_rows"),
            gather: sym!("ns_image_olm_gather"),
            seg_attn: sym!("ns_image_olm_seg_attn"),
            q8r_gemv: sym!("ns_image_olm_q8r_gemv"),
            q8r_half: sym!("ns_image_olm_q8r_half"),
            attn1_scratch: sym!("ns_image_olm_attn1_scratch"),
            attn1: sym!("ns_image_olm_attn1"),
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
