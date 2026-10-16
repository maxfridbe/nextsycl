//! The kernels this engine runs, bound from the audio kernel library by name: the backbone's and depth decoder's
//! cached decode on the MiniMax Music engine's (`kernels/audio/minimaxmusic3/mm3.h`: small-batch products, attention
//! against a half cache, Qwen3's q / k norm with RoPE, summed embeddings), the text encoder's attention and MLP and the
//! table-driven RoPE on its own (`kernels/audio/breezetts2/bz2.h`).

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
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub head_norm_rope: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, i64, i64, *const f32, f32, *const f32, i64) -> c_int,
    pub attn_window: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, i64, i64, i64, i64, i64, i64, i64, i64, f32, *mut f32) -> c_int,
    pub gelu_mul: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
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
                // SAFETY: the symbol is declared in mm3.h / bz2.h with the field's signature.
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
            add: sym!("ns_audio_mm3_add"),
            head_norm_rope: sym!("ns_audio_bz2_head_norm_rope"),
            attn_window: sym!("ns_audio_bz2_attn_window"),
            gelu_mul: sym!("ns_audio_bz2_gelu_mul"),
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
