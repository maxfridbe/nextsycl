//! This engine's own kernels (`kernels/audio/minimaxmusic3/mm3.h`), bound from the audio kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub gemv: unsafe extern "C" fn(G, *const f32, i64, i64, P, c_int, *const f32, i64, *const f32, *mut f32, i64, c_int) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub attn: unsafe extern "C" fn(G, *const f32, i64, P, P, i64, i64, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    pub kv_store: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, i64, i64, i64, i64, M, M) -> c_int,
    pub qk_norm_rope: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, i64, *const f32, f32, *const f32, i64) -> c_int,
    pub rope_partial: unsafe extern "C" fn(G, M, i64, i64, i64, i64, i64, i64, *const f32) -> c_int,
    pub embed: unsafe extern "C" fn(G, P, i64, *const i32, c_int, f32, *mut f32, i64, c_int) -> c_int,
    pub mix: unsafe extern "C" fn(G, *const f32, c_int, i64, *const f32, f32, *mut f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub bf16_to_half: unsafe extern "C" fn(G, *const u16, M, i64) -> c_int,
    pub bf16_to_float: unsafe extern "C" fn(G, *const u16, *mut f32, i64) -> c_int,
    pub quant_rows: unsafe extern "C" fn(G, *const f32, i64, i64, *mut i8, *mut f32) -> c_int,
    pub dequant_rows: unsafe extern "C" fn(G, *const i8, *const f32, i64, i64, M) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub dit_in: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, *mut f32) -> c_int,
    pub cfg_step: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, i64, f32, f32) -> c_int,
    pub blend: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, i64, f32, f32) -> c_int,
    pub nearest_rows: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32) -> c_int,
    pub transpose: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
    pub tanh: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
    pub conv1d: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, i64, *mut f32, i64) -> c_int,
    pub conv_transpose1d: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, *mut f32, i64) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the audio library, rebuilt: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in mm3.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            gemv: sym!("ns_audio_mm3_gemv"),
            attn_scratch: sym!("ns_audio_mm3_attn_scratch"),
            attn: sym!("ns_audio_mm3_attn"),
            kv_store: sym!("ns_audio_mm3_kv_store"),
            qk_norm_rope: sym!("ns_audio_mm3_qk_norm_rope"),
            rope_partial: sym!("ns_audio_mm3_rope_partial"),
            embed: sym!("ns_audio_mm3_embed"),
            mix: sym!("ns_audio_mm3_mix"),
            to_half: sym!("ns_audio_mm3_to_half"),
            bf16_to_half: sym!("ns_audio_mm3_bf16_to_half"),
            bf16_to_float: sym!("ns_audio_mm3_bf16_to_float"),
            quant_rows: sym!("ns_audio_mm3_quant_rows"),
            dequant_rows: sym!("ns_audio_mm3_dequant_rows"),
            add: sym!("ns_audio_mm3_add"),
            dit_in: sym!("ns_audio_mm3_dit_in"),
            cfg_step: sym!("ns_audio_mm3_cfg_step"),
            blend: sym!("ns_audio_mm3_blend"),
            nearest_rows: sym!("ns_audio_mm3_nearest_rows"),
            transpose: sym!("ns_audio_mm3_transpose"),
            tanh: sym!("ns_audio_mm3_tanh"),
            conv1d: sym!("ns_audio_mm3_conv1d"),
            conv_transpose1d: sym!("ns_audio_mm3_conv_transpose1d"),
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
