//! The kernels this engine runs, bound from the audio kernel library by name: its own (`kernels/audio/haru/haru.h`:
//! the language model's token rows, the Gated DeltaNet's convolution, recurrence and gated norm, attention's q / k norm
//! with partial RoPE, the split attention and its gated merge), the VibeVoice engine's products, norms and codec
//! blocks (`kernels/audio/vibevoice/vv.h`: the same causal VibeVoice codec), the MiniMax Music engine's add
//! (`kernels/audio/minimaxmusic3/mm3.h`) and the 12 Hz codec's SiLU (`kernels/audio/qwen3tts/q3t.h`). The shared ones
//! are called, never changed (CONTRIBUTING rule 6).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    // haru.h
    pub embed: unsafe extern "C" fn(G, P, i64, *const i32, i64, *mut f32, i64) -> c_int,
    pub gdn_conv: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, *const f32, *mut f32, *mut f32) -> c_int,
    pub gdn_rule: unsafe extern "C" fn(G, *const f32, i64, *const f32, i64, *const f32, *const f32, *mut f32, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub gated_norm: unsafe extern "C" fn(G, *const f32, *const f32, i64, *const f32, f32, i64, i64, i64, *mut f32) -> c_int,
    pub attn_prep: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, i64, i64, *const f32, *const f32, f32, *const f32, i64, *mut f32, M, M, i64) -> c_int,
    pub attn: unsafe extern "C" fn(G, *const f32, P, P, i64, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub attn_merge: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, *const f32, i64, i64, *mut f32) -> c_int,
    // vv.h
    pub gemv: unsafe extern "C" fn(G, *const f32, i64, i64, i64, P, c_int, P, i64, *const f32, c_int, *const f32, i64, *mut f32, i64) -> c_int,
    pub rms_norm: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, f32, *const f32, *const f32, i64, *mut f32) -> c_int,
    pub gemm: unsafe extern "C" fn(G, *const f32, P, P, i64, i64, i64, P, c_int, i64, *const f32, c_int, *const f32, i64, *mut f32, i64, *mut f32, i64) -> c_int,
    pub gemm_scratch: unsafe extern "C" fn(i64, i64, i64) -> i64,
    pub split: unsafe extern "C" fn(G, *const f32, i64, c_int, M, M) -> c_int,
    pub norm_hist: unsafe extern "C" fn(G, *const f32, i64, *const f32, i64, i64, i64, *const f32, f32, *mut f32) -> c_int,
    pub block_mix: unsafe extern "C" fn(G, *const f32, *mut f32, i64, i64, i64, *const f32, *const f32, *const f32, *const f32, f32, *mut f32) -> c_int,
    pub shift: unsafe extern "C" fn(G, *mut f32, i64, i64, i64) -> c_int,
    // mm3.h, q3t.h
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub silu: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
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
                // SAFETY: the symbol is declared in haru.h / vv.h / mm3.h / q3t.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            embed: sym!("ns_audio_haru_embed"),
            gdn_conv: sym!("ns_audio_haru_gdn_conv"),
            gdn_rule: sym!("ns_audio_haru_gdn_rule"),
            gated_norm: sym!("ns_audio_haru_gated_norm"),
            attn_prep: sym!("ns_audio_haru_attn_prep"),
            attn: sym!("ns_audio_haru_attn"),
            attn_merge: sym!("ns_audio_haru_attn_merge"),
            gemv: sym!("ns_audio_vv_gemv"),
            rms_norm: sym!("ns_audio_vv_rms_norm"),
            gemm: sym!("ns_audio_vv_gemm"),
            gemm_scratch: sym!("ns_audio_vv_gemm_scratch"),
            split: sym!("ns_audio_vv_split"),
            norm_hist: sym!("ns_audio_vv_norm_hist"),
            block_mix: sym!("ns_audio_vv_block_mix"),
            shift: sym!("ns_audio_vv_shift"),
            add: sym!("ns_audio_mm3_add"),
            silu: sym!("ns_audio_q3t_silu"),
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
