//! The kernels this engine runs, bound from the audio kernel library by name: the MiniMax Music engine's
//! product, key / value store, conversions, transpose and oneDNN convolutions (`kernels/audio/minimaxmusic3/mm3.h`),
//! Breeze TTS 2's per-head RoPE from a frequency table (`kernels/audio/breezetts2/bz2.h`), the 12 Hz codec's
//! layer-scaled residual and SiLU (`kernels/audio/qwen3tts/q3t.h`), and its own (`kernels/audio/vibevoice/vv.h`): the
//! language model's attention for its six query heads a key head, the tokenizers' streaming depthwise convolution.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub kv_store: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, i64, i64, i64, i64, M, M) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub transpose: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
    pub conv1d_lr: unsafe extern "C" fn(G, c_int, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, i64, i64, *mut f32, i64) -> c_int,
    pub head_norm_rope: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, i64, i64, *const f32, f32, *const f32, i64) -> c_int,
    pub scale_add: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, i64, i64) -> c_int,
    pub silu: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub attn: unsafe extern "C" fn(G, *const f32, i64, P, P, i64, i64, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    pub dwconv: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, *const f32, *const f32, i64, *mut f32) -> c_int,
    pub ctx_push: unsafe extern "C" fn(G, *mut f32, i64, *const f32, i64, i64) -> c_int,
    pub gemv: unsafe extern "C" fn(G, *const f32, i64, i64, i64, P, c_int, P, i64, *const f32, c_int, *const f32, *mut f32, i64) -> c_int,
    pub rms_norm: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, f32, *const f32, *const f32, i64, *mut f32) -> c_int,
    pub rope_kv: unsafe extern "C" fn(G, *mut f32, i64, i64, *const i32, i64, i64, i64, *const f32, M, M, i64, M, M, i64) -> c_int,
    pub attn_dec: unsafe extern "C" fn(G, *const f32, i64, i64, *const i32, P, P, i64, P, P, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    pub dequant_q8: unsafe extern "C" fn(G, P, P, i64, i64, M) -> c_int,
    pub swiglu_pairs: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
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
                // SAFETY: the symbol is declared in mm3.h / bz2.h / q3t.h / vv.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            kv_store: sym!("ns_audio_mm3_kv_store"),
            to_half: sym!("ns_audio_mm3_to_half"),
            add: sym!("ns_audio_mm3_add"),
            transpose: sym!("ns_audio_mm3_transpose"),
            conv1d_lr: sym!("ns_audio_mm3_conv1d_lr"),
            head_norm_rope: sym!("ns_audio_bz2_head_norm_rope"),
            scale_add: sym!("ns_audio_q3t_scale_add"),
            silu: sym!("ns_audio_q3t_silu"),
            attn_scratch: sym!("ns_audio_vv_attn_scratch"),
            attn: sym!("ns_audio_vv_attn"),
            dwconv: sym!("ns_audio_vv_dwconv"),
            ctx_push: sym!("ns_audio_vv_ctx_push"),
            gemv: sym!("ns_audio_vv_gemv"),
            rms_norm: sym!("ns_audio_vv_rms_norm"),
            rope_kv: sym!("ns_audio_vv_rope_kv"),
            attn_dec: sym!("ns_audio_vv_attn_dec"),
            dequant_q8: sym!("ns_audio_vv_dequant_q8"),
            swiglu_pairs: sym!("ns_audio_vv_swiglu_pairs"),
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
