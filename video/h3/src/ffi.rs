//! The silo (kernels/video/h3/silo/silo.h, `dist/silo/libnextsycl-h3.so`): the indexed row copy MPEG-Diffusion's
//! two-track attention assembles the full-resolution queries' keys and values with (crates/h3/src/mpeg.rs). Without
//! the file (or with NS_SILO=0) the two-track steps copy one row at a time through the shared kernels (slower, the same
//! bits).

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use nextsycl_core::Gpu;
use nextsycl_h3::mpeg::GatherFn;
use nextsycl_h3::Error;

/// The silo's name (`kernels/video/h3/silo` -> `dist/silo/libnextsycl-h3.so`)
pub const SILO: &str = "h3";

#[allow(clippy::type_complexity)]
pub struct Silo {
    pub gather_supported: unsafe extern "C" fn(i64, i64, i64) -> i32,
    pub gather: unsafe extern "C" fn(*mut c_void, *mut c_void, i64, *const c_void, i64, *const i32, i64, i64, i64) -> i32,
}

/// The silo, when built (None: NS_SILO=0, or no file)
// each field's declared type is the annotation: the transmute target is inferred from it
#[allow(clippy::missing_transmute_annotations)]
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static S: OnceLock<Result<Option<Silo>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
        macro_rules! sym {
            ($n:literal) => {{
                let p = lib.symbol($n);
                if p.is_null() {
                    return Err(format!("{} is not in the {SILO} silo", $n));
                }
                // SAFETY: the symbol is declared in silo.h with the field's signature.
                unsafe { std::mem::transmute::<*mut c_void, _>(p) }
            }};
        }
        Ok(Some(Silo { gather_supported: sym!("ns_video_h3_silo_gather_supported"), gather: sym!("ns_video_h3_silo_gather") }))
    })
    .as_ref()
    .map(Option::as_ref)
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The blocks' gather on the silo; the engine's device must share `gpu`'s queue (`Device::on_queue`)
pub fn gather(gpu: &Arc<Gpu>, s: &'static Silo) -> GatherFn {
    let gpu = gpu.clone();
    Box::new(move |dst, ds, src, ss, idx, rows, width, src_rows| {
        // SAFETY: plain query of sizes.
        if unsafe { (s.gather_supported)(width as i64, ss as i64, ds as i64) } == 0 {
            return Ok(false);
        }
        // SAFETY: the caller's buffers hold `rows` rows of `ds` and `src_rows` rows of `ss` 16-bit values, idx `rows` int32
        // on the card; one queue with the device
        let rc = unsafe { (s.gather)(gpu.raw(), dst, ds as i64, src, ss as i64, idx, rows as i64, width as i64, src_rows as i64) };
        if rc != 0 {
            return Err(Error(format!("the silo's gather: {}", nextsycl_core::api().map(|a| a.error()).unwrap_or_default())));
        }
        Ok(true)
    })
}
