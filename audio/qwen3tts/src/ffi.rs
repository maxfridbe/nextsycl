//! The kernels this engine runs, bound from the audio kernel library by name: the language models' decode on the
//! MiniMax Music engine's (`kernels/audio/minimaxmusic3/mm3.h`: the same Qwen3 layer shapes - cached attention, q / k
//! norm with rotary positions, small-batch products) and SiLU from its own (`kernels/audio/qwen3tts/q3t.h`, whose
//! other kernels are the 12 Hz codec's: nextsycl-codec12hz binds them).

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
    pub embed: unsafe extern "C" fn(G, P, i64, *const i32, c_int, f32, *mut f32, i64, c_int) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub bf16_to_float: unsafe extern "C" fn(G, *const u16, *mut f32, i64) -> c_int,
    pub quant_rows: unsafe extern "C" fn(G, *const f32, i64, i64, *mut i8, *mut f32) -> c_int,
    pub dequant_rows: unsafe extern "C" fn(G, *const i8, *const f32, i64, i64, M) -> c_int,
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
                // SAFETY: the symbol is declared in mm3.h / q3t.h with the field's signature.
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
            embed: sym!("ns_audio_mm3_embed"),
            to_half: sym!("ns_audio_mm3_to_half"),
            bf16_to_float: sym!("ns_audio_mm3_bf16_to_float"),
            quant_rows: sym!("ns_audio_mm3_quant_rows"),
            dequant_rows: sym!("ns_audio_mm3_dequant_rows"),
            add: sym!("ns_audio_mm3_add"),
            silu: sym!("ns_audio_q3t_silu"),
        })
    })
    .as_ref()
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The silo's kernels (kernels/audio/qwen3tts/silo/q3s.h, dist/silo/libnextsycl-qwen3tts.so), when it is there: the
/// frame on the device as one graph
#[allow(clippy::type_complexity)]
pub struct Silo {
    pub path: std::path::PathBuf,
    pub supported: unsafe extern "C" fn(i64, i64, i64) -> c_int,
    pub gemv: unsafe extern "C" fn(G, *const f32, i64, i64, P, c_int, *const f32, i64, *const f32, *mut f32, i64, c_int, *const f32, f32) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64, i64) -> i64,
    pub attn: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32, *const f32, f32, *const f32, *const i32, M, M, i64, i64, *mut f32, *mut f32,
                                   *mut i32) -> c_int,
    pub attn_small: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, i64, i64, *const f32, *const f32, f32, *const f32, M, M, i64, *mut f32) -> c_int,
    pub draw: unsafe extern "C" fn(G, *const f32, i64, c_int, P, M, *mut u8, *const i32, *const f32, i64, i64, *mut f32) -> c_int,
    pub frame_in: unsafe extern "C" fn(G, P, i64, i64, i64, P, P, *const f32, *mut f32) -> c_int,
    pub advance: unsafe extern "C" fn(G, M) -> c_int,
    pub graph_begin: unsafe extern "C" fn(G, *mut M) -> c_int,
    pub graph_end: unsafe extern "C" fn(G, M, *mut M) -> c_int,
    pub graph_launch: unsafe extern "C" fn(G, M) -> c_int,
    pub graph_free: unsafe extern "C" fn(M),
    pub read_async: unsafe extern "C" fn(G, M, P, usize) -> c_int,
    pub quant_q8: unsafe extern "C" fn(G, *const f32, i64, i64, *mut i8, M) -> c_int,
    pub dequant_q8: unsafe extern "C" fn(G, *const i8, P, i64, i64, M) -> c_int,
}

// SAFETY: function pointers into the silo library, which stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo (None: the shared kernels, the draws on the host)
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static SILO: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    SILO.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo("qwen3tts").map_err(|e| e.0)? else { return Ok(None) };
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{}: no {} (rebuild it: ./build.sh kernels)", lib.path.display(), $name));
                }
                // SAFETY: the symbol is declared in q3s.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Some(Silo {
            path: lib.path.clone(),
            supported: sym!("ns_q3s_supported"),
            gemv: sym!("ns_q3s_gemv"),
            attn_scratch: sym!("ns_q3s_attn_scratch"),
            attn: sym!("ns_q3s_attn"),
            attn_small: sym!("ns_q3s_attn_small"),
            draw: sym!("ns_q3s_draw"),
            frame_in: sym!("ns_q3s_frame_in"),
            advance: sym!("ns_q3s_advance"),
            graph_begin: sym!("ns_q3s_graph_begin"),
            graph_end: sym!("ns_q3s_graph_end"),
            graph_launch: sym!("ns_q3s_graph_launch"),
            graph_free: sym!("ns_q3s_graph_free"),
            read_async: sym!("ns_q3s_read_async"),
            quant_q8: sym!("ns_q3s_quant_q8"),
            dequant_q8: sym!("ns_q3s_dequant_q8"),
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
