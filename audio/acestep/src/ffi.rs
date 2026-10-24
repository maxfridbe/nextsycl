//! The kernels of the audio library this engine needs beyond the shared diffusion ones (nsd): its own
//! (`kernels/audio/acestep/ace.h`: the windowed bidirectional attention, Oobleck's snake) and the MiniMax Music
//! engine's float32 transpose (`kernels/audio/minimaxmusic3/mm3.h`), which turns the VAE's channels-first output into
//! frames.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub attn_window: unsafe extern "C" fn(G, P, i64, P, P, i64, i64, i64, i64, i64, f32, M, i64) -> c_int,
    pub attn_segments: unsafe extern "C" fn(G, P, i64, P, P, i64, i64, i64, i64, i64, f32, M, i64) -> c_int,
    pub snake: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, *mut f32) -> c_int,
    pub transpose: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
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
                // SAFETY: the symbol is declared in ace.h / mm3.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api { attn_window: sym!("ns_audio_ace_attn_window"), attn_segments: sym!("ns_audio_ace_attn_segments"), snake: sym!("ns_audio_ace_snake"), transpose: sym!("ns_audio_mm3_transpose") })
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
