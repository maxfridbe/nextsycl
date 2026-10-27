//! This engine's own kernels (`kernels/image/mossimage/mo.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub situ: unsafe extern "C" fn(G, *const f32, i64, *const f32, i64, i64, i64, M, i64) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    pub cfg: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, i64, f32) -> c_int,
    // the text encoder's (te.cpp)
    pub gdn_gates: unsafe extern "C" fn(G, *const f32, *mut f32, *const f32, *const f32, *mut f32, i64, i64, i64) -> c_int,
    pub conv_silu: unsafe extern "C" fn(G, *const f32, *mut f32, *const f32, *mut f32, i64, i64, c_int) -> c_int,
    pub expand: unsafe extern "C" fn(G, *const f32, i64, *mut f32, i64, i64, i64, i64) -> c_int,
    pub l2_norm: unsafe extern "C" fn(G, *mut f32, i64, i64, f32) -> c_int,
    pub kda_scan: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, i64, i64) -> c_int,
    pub gdn_out: unsafe extern "C" fn(G, *const f32, *const f32, i64, *const f32, *mut f32, i64, i64, i64, f32) -> c_int,
    pub qk_norm_rope: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32, i64, i64, i64, *const f32, f32, i64, f32) -> c_int,
    pub gate_mul: unsafe extern "C" fn(G, *mut f32, *const f32, i64, i64, i64, i64) -> c_int,
    pub silu_mul: unsafe extern "C" fn(G, *const f32, *const f32, M, i64) -> c_int,
    // the VAE's (vae.cpp)
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
                // SAFETY: the symbol is declared in mo.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            situ: sym!("ns_image_mo_situ"),
            to_half: sym!("ns_image_mo_to_half"),
            to_float: sym!("ns_image_mo_to_float"),
            axpy: sym!("ns_image_mo_axpy"),
            cfg: sym!("ns_image_mo_cfg"),
            gdn_gates: sym!("ns_image_mo_gdn_gates"),
            conv_silu: sym!("ns_image_mo_conv_silu"),
            expand: sym!("ns_image_mo_expand"),
            l2_norm: sym!("ns_image_mo_l2_norm"),
            kda_scan: sym!("ns_image_mo_kda_scan"),
            gdn_out: sym!("ns_image_mo_gdn_out"),
            qk_norm_rope: sym!("ns_image_mo_qk_norm_rope"),
            gate_mul: sym!("ns_image_mo_gate_mul"),
            silu_mul: sym!("ns_image_mo_silu_mul"),
            group_norm: sym!("ns_image_mo_group_norm"),
            softmax_rows: sym!("ns_image_mo_softmax_rows"),
            up2: sym!("ns_image_mo_up2"),
            to_rgb8: sym!("ns_image_mo_to_rgb8"),
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
