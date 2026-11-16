//! The kernels this engine runs, bound from the audio kernel library by name: its own (`kernels/audio/omnivoice/ov.h`:
//! the audio rows' embeddings, the grouped heads laid out for the attention, the step's scores, the codec encoder's
//! per-channel norm and ELU), and the MiniMax Music engine's convolutions on oneDNN, float32 to half and adds
//! (`kernels/audio/minimaxmusic3/mm3.h`). The linear layers, norms, rotary positions, attention and Snake are the
//! shared diffusion kernels (nextsycl_diffusion::kernels).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type P = *const c_void;
type M = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub audio_embed: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const i32, *const i32, i64, *mut f32, i64) -> c_int,
    pub expand_qkv: unsafe extern "C" fn(G, P, i64, i64, i64, i64, M) -> c_int,
    pub score: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, f32, *mut i32, *mut f32) -> c_int,
    pub chan_norm: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, f32, *mut f32) -> c_int,
    pub elu: unsafe extern "C" fn(G, *const f32, i64, *mut f32) -> c_int,
    pub conv1d_lr: unsafe extern "C" fn(G, c_int, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, i64, i64, *mut f32, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub transpose: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
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
                // SAFETY: the symbol is declared in ov.h / mm3.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            audio_embed: sym!("ns_audio_ov_audio_embed"),
            expand_qkv: sym!("ns_audio_ov_expand_qkv"),
            score: sym!("ns_audio_ov_score"),
            chan_norm: sym!("ns_audio_ov_chan_norm"),
            elu: sym!("ns_audio_ov_elu"),
            conv1d_lr: sym!("ns_audio_mm3_conv1d_lr"),
            add: sym!("ns_audio_mm3_add"),
            transpose: sym!("ns_audio_mm3_transpose"),
            to_half: sym!("ns_audio_mm3_to_half"),
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
