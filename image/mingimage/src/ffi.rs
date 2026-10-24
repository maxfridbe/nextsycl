//! This engine's own kernels (`kernels/image/mingimage/mi.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub ln_mod: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const f32, M) -> c_int,
    pub rms_mod: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, f32, *const f32, M) -> c_int,
    pub softmax_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M) -> c_int,
    pub silu: unsafe extern "C" fn(G, M, i64) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    pub to_rgba8: unsafe extern "C" fn(G, P, i64, i64, *mut u8) -> c_int,
    pub gather_rows: unsafe extern "C" fn(G, P, i64, *const i32, i64, M) -> c_int,
    pub combine: unsafe extern "C" fn(G, *mut f32, i64, i64, P, *const i32, *const f32, i64) -> c_int,
    pub gate_add: unsafe extern "C" fn(G, *mut f32, i64, i64, P, *const f32) -> c_int,
    pub swiglu_scaled: unsafe extern "C" fn(G, P, i64, i64, f32, M) -> c_int,
    pub to_half_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M, i64) -> c_int,
    pub repeat_heads: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, M) -> c_int,
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
                // SAFETY: the symbol is declared in mi.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            ln_mod: sym!("ns_image_mi_ln_mod"),
            rms_mod: sym!("ns_image_mi_rms_mod"),
            softmax_rows: sym!("ns_image_mi_softmax_rows"),
            silu: sym!("ns_image_mi_silu"),
            axpy: sym!("ns_image_mi_axpy"),
            to_half: sym!("ns_image_mi_to_half"),
            to_float: sym!("ns_image_mi_to_float"),
            up2: sym!("ns_image_mi_up2"),
            to_rgba8: sym!("ns_image_mi_to_rgba8"),
            gather_rows: sym!("ns_image_mi_gather_rows"),
            combine: sym!("ns_image_mi_combine"),
            gate_add: sym!("ns_image_mi_gate_add"),
            swiglu_scaled: sym!("ns_image_mi_swiglu_scaled"),
            to_half_rows: sym!("ns_image_mi_to_half_rows"),
            repeat_heads: sym!("ns_image_mi_repeat_heads"),
        })
    })
    .as_ref()
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The scale the MLPs' gated products are made at in half (their next linear's float32 output multiplied back)
pub const ACT_SCALE: f32 = 1.0 / 256.0;

/// out [m, c] half = silu(x[:, :c]) * x[:, c:] * ACT_SCALE (x [m, 2c] half)
pub fn swiglu(gpu: &nextsycl_core::Gpu, x: P, m: usize, c: usize, out: M) -> nextsycl_core::Result<()> {
    let k = api()?;
    // SAFETY: x holds m x 2c halves, out m x c.
    check(unsafe { (k.swiglu_scaled)(gpu.raw(), x, m as i64, c as i64, ACT_SCALE, out) }, "swiglu")
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextsycl_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextsycl_core::Error(format!("{what}: {why}")))
}
