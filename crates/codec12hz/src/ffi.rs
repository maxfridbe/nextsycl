//! The codec's kernels, bound from the audio kernel library by name: its own (`kernels/audio/qwen3tts/q3t.h`) and
//! MiniMax Music's convolutions and copies (`kernels/audio/minimaxmusic3/mm3.h`).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub bf16_to_float: unsafe extern "C" fn(G, *const u16, *mut f32, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub transpose: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
    pub conv1d_lr: unsafe extern "C" fn(G, c_int, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, i64, i64, *mut f32, i64) -> c_int,
    pub snake_beta: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, *mut f32) -> c_int,
    pub dwconv: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, i64, *mut f32) -> c_int,
    pub window_attn: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub rope: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, i64, f32, i64) -> c_int,
    pub scale_add: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, i64, i64) -> c_int,
    pub clamp: unsafe extern "C" fn(G, *mut f32, i64, f32, f32) -> c_int,
    pub elu: unsafe extern "C" fn(G, *const f32, i64, *mut f32) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the 12 Hz codec runs on the audio library, rebuilt: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in mm3.h / q3t.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            bf16_to_float: sym!("ns_audio_mm3_bf16_to_float"),
            add: sym!("ns_audio_mm3_add"),
            transpose: sym!("ns_audio_mm3_transpose"),
            conv1d_lr: sym!("ns_audio_mm3_conv1d_lr"),
            snake_beta: sym!("ns_audio_q3t_snake_beta"),
            dwconv: sym!("ns_audio_q3t_dwconv"),
            window_attn: sym!("ns_audio_q3t_window_attn"),
            rope: sym!("ns_audio_q3t_rope"),
            scale_add: sym!("ns_audio_q3t_scale_add"),
            clamp: sym!("ns_audio_q3t_clamp"),
            elu: sym!("ns_audio_q3t_elu"),
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
