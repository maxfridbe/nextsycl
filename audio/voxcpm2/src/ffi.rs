//! The kernels this engine runs, bound from the audio kernel library by name: the language models' decode on the
//! MiniMax Music engine's (`kernels/audio/minimaxmusic3/mm3.h`: small-batch products, cached attention, conversions,
//! the oneDNN convolutions), the local transformers' attention, the long RoPE and the AudioVAE's own on this engine's
//! (`kernels/audio/voxcpm2/vcp.h`); SiLU from Qwen3-TTS's (`q3t.h`).

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
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub bf16_to_float: unsafe extern "C" fn(G, *const u16, *mut f32, i64) -> c_int,
    pub quant_rows: unsafe extern "C" fn(G, *const f32, i64, i64, *mut i8, *mut f32) -> c_int,
    pub dequant_rows: unsafe extern "C" fn(G, *const i8, *const f32, i64, i64, M) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub transpose: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
    pub conv1d_lr: unsafe extern "C" fn(G, c_int, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, i64, i64, *mut f32, i64) -> c_int,
    pub tanh: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
    pub silu: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
    pub rope: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, i64, i64, *const f32, i64) -> c_int,
    pub local_attn: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, i64, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub snake: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *mut f32) -> c_int,
    pub dwconv: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, i64, i64, *mut f32) -> c_int,
    pub affine: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32, *const f32) -> c_int,
    pub fsq: unsafe extern "C" fn(G, *mut f32, i64, f32) -> c_int,
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
                // SAFETY: the symbol is declared in mm3.h / q3t.h / vcp.h with the field's signature.
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
            to_half: sym!("ns_audio_mm3_to_half"),
            bf16_to_float: sym!("ns_audio_mm3_bf16_to_float"),
            quant_rows: sym!("ns_audio_mm3_quant_rows"),
            dequant_rows: sym!("ns_audio_mm3_dequant_rows"),
            add: sym!("ns_audio_mm3_add"),
            transpose: sym!("ns_audio_mm3_transpose"),
            conv1d_lr: sym!("ns_audio_mm3_conv1d_lr"),
            tanh: sym!("ns_audio_mm3_tanh"),
            silu: sym!("ns_audio_q3t_silu"),
            rope: sym!("ns_audio_vcp_rope"),
            local_attn: sym!("ns_audio_vcp_local_attn"),
            snake: sym!("ns_audio_vcp_snake"),
            dwconv: sym!("ns_audio_vcp_dwconv"),
            affine: sym!("ns_audio_vcp_affine"),
            fsq: sym!("ns_audio_vcp_fsq"),
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

/// The silo's kernels (kernels/audio/voxcpm2/silo/vcs.h, dist/silo/libnextsycl-voxcpm2.so), when it is there: a
/// patch's sampler on the device, recorded as one graph
pub struct Silo {
    pub supported: unsafe extern "C" fn(i64) -> c_int,
    pub cfg_euler: unsafe extern "C" fn(G, *mut f32, *const f32, *const i32, i64, f32, f32) -> c_int,
    pub graph_begin: unsafe extern "C" fn(G, *mut M) -> c_int,
    pub graph_end: unsafe extern "C" fn(G, M, *mut M) -> c_int,
    pub graph_launch: unsafe extern "C" fn(G, M) -> c_int,
    pub graph_free: unsafe extern "C" fn(M),
}

// SAFETY: function pointers into the silo library, which stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo (None: the sampler's update on the host, an evaluation at a time)
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static SILO: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    SILO.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo("voxcpm2").map_err(|e| e.0)? else { return Ok(None) };
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{}: no {} (rebuild it: ./build.sh kernels)", lib.path.display(), $name));
                }
                // SAFETY: the symbol is declared in vcs.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Some(Silo {
            supported: sym!("ns_vcs_supported"),
            cfg_euler: sym!("ns_vcs_cfg_euler"),
            graph_begin: sym!("ns_vcs_graph_begin"),
            graph_end: sym!("ns_vcs_graph_end"),
            graph_launch: sym!("ns_vcs_graph_launch"),
            graph_free: sym!("ns_vcs_graph_free"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| nextsycl_core::Error(e.clone()))
}
