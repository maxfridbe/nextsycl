//! The kernels this engine runs, bound from the audio kernel library by name: its own (`kernels/audio/vibevoice/vv.h`:
//! the products, the decode step, the head's and the tokenizers' kernels), and for a prompt's pass and the load the
//! MiniMax Music engine's key / value store, conversion and add (`kernels/audio/minimaxmusic3/mm3.h`), Breeze TTS 2's
//! per-head RoPE from a frequency table (`kernels/audio/breezetts2/bz2.h`) and the 12 Hz codec's SiLU
//! (`kernels/audio/qwen3tts/q3t.h`).

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
    pub head_norm_rope: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, i64, i64, *const f32, f32, *const f32, i64) -> c_int,
    pub silu: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub attn: unsafe extern "C" fn(G, *const f32, i64, P, P, i64, i64, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    pub gemv: unsafe extern "C" fn(G, *const f32, i64, i64, i64, P, c_int, P, i64, *const f32, c_int, *const f32, i64, *mut f32, i64) -> c_int,
    pub rms_norm: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, f32, *const f32, *const f32, i64, *mut f32) -> c_int,
    pub rope_kv: unsafe extern "C" fn(G, *mut f32, i64, i64, *const i32, i64, i64, i64, *const f32, M, M, i64, M, M, i64) -> c_int,
    pub attn_dec: unsafe extern "C" fn(G, *const f32, i64, i64, *const i32, P, P, i64, P, P, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    pub dequant_q8: unsafe extern "C" fn(G, P, P, i64, i64, M) -> c_int,
    pub swiglu_pairs: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
    pub gemm: unsafe extern "C" fn(G, *const f32, P, P, i64, i64, i64, P, c_int, i64, *const f32, c_int, *const f32, i64, *mut f32, i64, *mut f32, i64) -> c_int,
    pub gemm_scratch: unsafe extern "C" fn(i64, i64, i64) -> i64,
    pub split: unsafe extern "C" fn(G, *const f32, i64, c_int, M, M) -> c_int,
    pub norm_hist: unsafe extern "C" fn(G, *const f32, i64, *const f32, i64, i64, i64, *const f32, f32, *mut f32) -> c_int,
    pub block_mix: unsafe extern "C" fn(G, *const f32, *mut f32, i64, i64, i64, *const f32, *const f32, *const f32, *const f32, f32, *mut f32) -> c_int,
    pub shift: unsafe extern "C" fn(G, *mut f32, i64, i64, i64) -> c_int,
    pub silu_rows: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, *mut f32) -> c_int,
    pub dpm: unsafe extern "C" fn(G, *const f32, i64, f32, *mut f32, *mut f32, f32, f32, f32, f32, f32, c_int, *mut f32, i64, *mut f32, *mut f32, f32, f32) -> c_int,
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
            head_norm_rope: sym!("ns_audio_bz2_head_norm_rope"),
            silu: sym!("ns_audio_q3t_silu"),
            attn_scratch: sym!("ns_audio_vv_attn_scratch"),
            attn: sym!("ns_audio_vv_attn"),
            gemv: sym!("ns_audio_vv_gemv"),
            rms_norm: sym!("ns_audio_vv_rms_norm"),
            rope_kv: sym!("ns_audio_vv_rope_kv"),
            attn_dec: sym!("ns_audio_vv_attn_dec"),
            dequant_q8: sym!("ns_audio_vv_dequant_q8"),
            swiglu_pairs: sym!("ns_audio_vv_swiglu_pairs"),
            gemm: sym!("ns_audio_vv_gemm"),
            split: sym!("ns_audio_vv_split"),
            gemm_scratch: sym!("ns_audio_vv_gemm_scratch"),
            norm_hist: sym!("ns_audio_vv_norm_hist"),
            block_mix: sym!("ns_audio_vv_block_mix"),
            shift: sym!("ns_audio_vv_shift"),
            silu_rows: sym!("ns_audio_vv_silu_rows"),
            dpm: sym!("ns_audio_vv_dpm"),
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
