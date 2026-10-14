//! This engine's own C ABI (`kernels/image/unlimitedocr/uocr.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type F = *const f32;
type FM = *mut f32;

pub struct Api {
    pub to_half: unsafe extern "C" fn(G, F, *mut c_void, i64) -> c_int,
    pub quick_gelu: unsafe extern "C" fn(G, FM, i64) -> c_int,
    /// (img, B, H, W, P, out)
    pub patchify: unsafe extern "C" fn(G, F, i64, i64, i64, i64, FM) -> c_int,
    /// (x, B, H, W, C, stride, out)
    pub im2col3: unsafe extern "C" fn(G, F, i64, i64, i64, i64, i64, FM) -> c_int,
    /// (x, B, H, W, C, w, back, out)
    pub window: unsafe extern "C" fn(G, F, i64, i64, i64, i64, i64, c_int, FM) -> c_int,
    /// (qkv, B, Gh, Gw, heads, D, rh, rw, out)
    pub sam_attn: unsafe extern "C" fn(G, F, i64, i64, i64, i64, i64, F, F, FM) -> c_int,
    /// (Q5_0 blocks, n, half out)
    pub q5_0_half: unsafe extern "C" fn(G, *const c_void, i64, *mut c_void) -> c_int,
    /// (src, src stride, dst, dst stride, rows, cols)
    pub copy_rows: unsafe extern "C" fn(G, F, i64, FM, i64, i64, i64) -> c_int,
    /// (src, C, idx, n, out)
    pub gather: unsafe extern "C" fn(G, F, i64, *const i32, i64, FM) -> c_int,
    /// (dst, C, src, idx, w, n)
    pub scatter_add: unsafe extern "C" fn(G, FM, i64, F, *const i32, F, i64) -> c_int,
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
                // SAFETY: the symbol is declared in uocr.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            to_half: sym!("ns_image_uocr_to_half"),
            quick_gelu: sym!("ns_image_uocr_quick_gelu"),
            patchify: sym!("ns_image_uocr_patchify"),
            im2col3: sym!("ns_image_uocr_im2col3"),
            window: sym!("ns_image_uocr_window"),
            sam_attn: sym!("ns_image_uocr_sam_attn"),
            q5_0_half: sym!("ns_image_uocr_q5_0_half"),
            copy_rows: sym!("ns_image_uocr_copy_rows"),
            gather: sym!("ns_image_uocr_gather"),
            scatter_add: sym!("ns_image_uocr_scatter_add"),
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
