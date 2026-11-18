//! This engine's silo (`kernels/video/h3/silo/silo.h`, `dist/silo/libnextsycl-h3.so`), bound by name:
//! - the kernels of Open-H3-Regenerate-2K's fast video decoder (window attention as variable-length sequences on ARK's
//!   sycl-tla kernel, the per-block global tokens' attention, an indexed row copy), handed to the decoder
//!   (`nextsycl_h3::fastvae::install`); without the silo the regen2k job decodes with the H3 video decoder;
//! - the indexed row copy MPEG-Diffusion's two-track attention assembles the full-resolution queries' keys and values
//!   with (crates/h3/src/mpeg.rs); without it (or with NS_SILO=0) the two-track steps copy one row at a time through
//!   the shared kernels (slower, the same bits).
//!
//! Every call takes the GPU's SYCL queue and reports failures through `ns_video_h3_silo_error`.

use std::ffi::{c_char, c_void, CStr};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use nextsycl_core::Gpu;
use nextsycl_h3::fastvae::Kernels;
use nextsycl_h3::mpeg::GatherFn;
use nextsycl_h3::Error;

/// The silo's name (`kernels/video/h3/silo` -> `dist/silo/libnextsycl-h3.so`)
pub const SILO: &str = "h3";

macro_rules! sym {
    ($lib:expr, $name:literal) => {{
        let p = $lib.symbol($name);
        if p.is_null() {
            return Err(format!("{}: no {}", $lib.path.display(), $name));
        }
        // SAFETY: the symbol is declared in silo.h with the field's signature.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            std::mem::transmute::<*mut c_void, _>(p)
        }
    }};
}

/// Binds the silo and installs its kernels for the fast decoder: its path, or None when there is none (NS_SILO=0, not
/// built)
pub fn install() -> Result<Option<PathBuf>, String> {
    let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
    nextsycl_h3::fastvae::install(Kernels {
        error: sym!(lib, "ns_video_h3_silo_error"),
        supported: sym!(lib, "ns_video_h3_silo_supported"),
        rows: sym!(lib, "ns_video_h3_silo_rows"),
        varlen: sym!(lib, "ns_video_h3_silo_varlen"),
        glob_init: sym!(lib, "ns_video_h3_silo_glob_init"),
        glob: sym!(lib, "ns_video_h3_silo_glob"),
        glob_out: sym!(lib, "ns_video_h3_silo_glob_out"),
    });
    Ok(Some(lib.path.clone()))
}

/// MPEG-Diffusion's part of the silo
#[allow(clippy::type_complexity)]
pub struct Mpeg {
    pub error: unsafe extern "C" fn() -> *const c_char,
    pub gather_supported: unsafe extern "C" fn(i64, i64, i64) -> i32,
    pub gather: unsafe extern "C" fn(*mut c_void, *mut c_void, i64, *const c_void, i64, *const i32, i64, i64, i64) -> i32,
}

/// MPEG-Diffusion's part of the silo, when built (None: NS_SILO=0, or no file)
pub fn mpeg() -> nextsycl_core::Result<Option<&'static Mpeg>> {
    static S: OnceLock<Result<Option<Mpeg>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
        Ok(Some(Mpeg {
            error: sym!(lib, "ns_video_h3_silo_error"),
            gather_supported: sym!(lib, "ns_video_h3_silo_gather_supported"),
            gather: sym!(lib, "ns_video_h3_silo_gather"),
        }))
    })
    .as_ref()
    .map(Option::as_ref)
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The blocks' gather on the silo; the engine's device must share `gpu`'s queue (`Device::on_queue`)
pub fn gather(gpu: &Arc<Gpu>, s: &'static Mpeg) -> GatherFn {
    let gpu = gpu.clone();
    Box::new(move |dst, ds, src, ss, idx, rows, width, src_rows| {
        // SAFETY: plain query of sizes.
        if unsafe { (s.gather_supported)(width as i64, ss as i64, ds as i64) } == 0 {
            return Ok(false);
        }
        // SAFETY: the caller's buffers hold `rows` rows of `ds` and `src_rows` rows of `ss` 16-bit values, idx `rows` int32
        // on the card; one queue with the device
        let rc = unsafe { (s.gather)(gpu.queue(), dst, ds as i64, src, ss as i64, idx, rows as i64, width as i64, src_rows as i64) };
        if rc != 0 {
            // SAFETY: the silo's error is a NUL-terminated string it keeps (thread-local, until its next failure)
            let why = unsafe { CStr::from_ptr((s.error)()) }.to_string_lossy().into_owned();
            return Err(Error(format!("the silo's gather: {why}")));
        }
        Ok(true)
    })
}
