//! The kernels of the audio library this engine needs beyond the shared diffusion ones (nsd): its own
//! (`kernels/audio/yue2/yue.h`: the Q8_0 decode product and embedding rows, the attention's inputs and the decode's
//! attention, the flow stage's query heads and element-wise steps) ACE-Step's SnakeBeta (`kernels/audio/acestep/ace.h`: Oobleck's
//! activation, YuE2-Vae's too), MiniMax Music 3's oneDNN convolutions (`kernels/audio/minimaxmusic3/mm3.h`: strict float32; nsd_conv1d is a plain loop) and SheetSage2's encoder's three (`kernels/audio/yue2/ss2.cpp`: the depthwise
//! convolutions, the global response norm, the GLU).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub gemv_q8: unsafe extern "C" fn(G, *const f32, i64, i64, P, i64, *mut f32, i64, c_int) -> c_int,
    pub embed_q8: unsafe extern "C" fn(G, P, i64, *const i32, i64, *mut f32) -> c_int,
    pub qkv: unsafe extern "C" fn(G, *const f32, i64, *const f32, *const f32, f32, *const f32, i64, M, M, M, i64) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64) -> i64,
    pub attn: unsafe extern "C" fn(G, P, P, P, i64, *mut f32, *mut f32) -> c_int,
    pub heads: unsafe extern "C" fn(G, P, M, i64, c_int) -> c_int,
    pub add_vec_rows: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32, *const f32) -> c_int,
    pub linear_f32: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, i64, *const f32, *mut f32) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, i64, f32) -> c_int,
    pub conv1d: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, i64, *mut f32, i64) -> c_int,
    pub conv_transpose1d: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, *mut f32, i64) -> c_int,
    pub snake: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, *mut f32) -> c_int,
    pub dwconv: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, i64, *const f32, *mut f32) -> c_int,
    pub grn: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, M, *mut f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, i64, M) -> c_int,
    pub to_f32: unsafe extern "C" fn(G, P, i64, *mut f32) -> c_int,
    pub glu: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the audio library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in yue.h / ace.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            gemv_q8: sym!("ns_audio_yue2_gemv_q8"),
            embed_q8: sym!("ns_audio_yue2_embed_q8"),
            qkv: sym!("ns_audio_yue2_qkv"),
            attn_scratch: sym!("ns_audio_yue2_attn_scratch"),
            attn: sym!("ns_audio_yue2_attn"),
            heads: sym!("ns_audio_yue2_heads"),
            add_vec_rows: sym!("ns_audio_yue2_add_vec_rows"),
            linear_f32: sym!("ns_audio_yue2_linear_f32"),
            axpy: sym!("ns_audio_yue2_axpy"),
            conv1d: sym!("ns_audio_mm3_conv1d"),
            conv_transpose1d: sym!("ns_audio_mm3_conv_transpose1d"),
            snake: sym!("ns_audio_ace_snake"),
            dwconv: sym!("ns_audio_yue2_dwconv"),
            grn: sym!("ns_audio_yue2_grn"),
            to_half: sym!("ns_audio_yue2_to_half"),
            to_f32: sym!("ns_audio_yue2_to_f32"),
            glu: sym!("ns_audio_yue2_glu"),
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

/// YuE2's silo (`kernels/audio/yue2/silo` -> `dist/silo/libnextsycl-yue2.so`): its Q8_0 matrices repacked (silo.h)
pub struct Silo {
    pub supported: unsafe extern "C" fn(G, i64) -> c_int,
    pub scales_at: unsafe extern "C" fn(i64, i64) -> usize,
    pub bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub pack: unsafe extern "C" fn(G, P, i64, i64, M) -> c_int,
    pub gemv: unsafe extern "C" fn(G, *const f32, i64, i64, P, P, i64, *mut f32, i64, c_int) -> c_int,
    pub half: unsafe extern "C" fn(G, P, P, i64, i64, M) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo, when built (None: NS_SILO=0, or no file - the matrices as stored, the engine's own kernels)
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static S: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo("yue2").map_err(|e| e.0)? else { return Ok(None) };
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{}: no {} (rebuild it: ./build.sh kernels)", lib.path.display(), $name));
                }
                // SAFETY: the symbol is declared in silo.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Some(Silo {
            supported: sym!("ns_audio_yue2_silo_supported"),
            scales_at: sym!("ns_audio_yue2_silo_scales_at"),
            bytes: sym!("ns_audio_yue2_silo_bytes"),
            pack: sym!("ns_audio_yue2_silo_pack"),
            gemv: sym!("ns_audio_yue2_silo_gemv"),
            half: sym!("ns_audio_yue2_silo_half"),
        }))
    })
    .as_ref()
    .map(|o| o.as_ref())
    .map_err(|e| nextsycl_core::Error(e.clone()))
}
