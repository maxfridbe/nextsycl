//! The kernels of the audio library this engine needs beyond the shared diffusion ones (nsd): its own
//! (`kernels/audio/yue2/yue.h`: the Q8_0 decode product and embedding rows, the attention's inputs and the decode's
//! attention, the flow stage's query heads and element-wise steps) and ACE-Step's SnakeBeta (`kernels/audio/acestep/ace.h`: Oobleck's
//! activation, YuE2-Vae's too) and SheetSage2's encoder's three (`kernels/audio/yue2/ss2.cpp`: the depthwise
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
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, i64, f32) -> c_int,
    pub snake: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, *mut f32) -> c_int,
    pub dwconv: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, i64, *const f32, *mut f32) -> c_int,
    pub grn: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, M, *mut f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, i64, M) -> c_int,
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
            axpy: sym!("ns_audio_yue2_axpy"),
            snake: sym!("ns_audio_ace_snake"),
            dwconv: sym!("ns_audio_yue2_dwconv"),
            grn: sym!("ns_audio_yue2_grn"),
            to_half: sym!("ns_audio_yue2_to_half"),
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
