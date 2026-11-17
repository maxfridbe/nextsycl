//! This engine's silo (`kernels/video/h3/silo/silo.h`, `dist/silo/libnextsycl-h3.so`): the kernels of
//! Open-H3-Regenerate-2K's fast video decoder (window attention as variable-length sequences on ARK's sycl-tla kernel,
//! the per-block global tokens' attention, an indexed row copy), bound by name and handed to the decoder
//! (`nextsycl_h3::fastvae::install`). Without the silo the regen2k job decodes with the H3 video decoder.

use std::path::PathBuf;

use nextsycl_h3::fastvae::Kernels;

/// The silo's name (`kernels/video/h3/silo` -> `dist/silo/libnextsycl-h3.so`)
const SILO: &str = "h3";

/// Binds the silo and installs its kernels for the fast decoder: its path, or None when there is none (NS_SILO=0, not
/// built)
pub fn install() -> Result<Option<PathBuf>, String> {
    let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
    macro_rules! sym {
        ($name:literal) => {{
            let p = lib.symbol($name);
            if p.is_null() {
                return Err(format!("{}: no {}", lib.path.display(), $name));
            }
            // SAFETY: the symbol is declared in silo.h with the field's signature.
            #[allow(clippy::missing_transmute_annotations)]
            unsafe {
                std::mem::transmute::<*mut std::ffi::c_void, _>(p)
            }
        }};
    }
    nextsycl_h3::fastvae::install(Kernels {
        error: sym!("ns_video_h3_silo_error"),
        supported: sym!("ns_video_h3_silo_supported"),
        rows: sym!("ns_video_h3_silo_rows"),
        varlen: sym!("ns_video_h3_silo_varlen"),
        glob_init: sym!("ns_video_h3_silo_glob_init"),
        glob: sym!("ns_video_h3_silo_glob"),
        glob_out: sym!("ns_video_h3_silo_glob_out"),
    });
    Ok(Some(lib.path.clone()))
}
