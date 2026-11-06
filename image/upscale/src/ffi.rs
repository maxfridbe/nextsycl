//! The upscalers' own kernels (`kernels/image/upscale/up.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{Error, Gpu, Result};

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub pre: unsafe extern "C" fn(G, *const u8, i64, i64, i64, i64, i64, i64, i64, c_int, M) -> c_int,
    pub pre_alpha: unsafe extern "C" fn(G, *const u8, i64, i64, i64, i64, i64, i64, M) -> c_int,
    pub post: unsafe extern "C" fn(G, P, i64, i64, i64, P, *mut u8, i64, i64, i64, i64, i64, i64) -> c_int,
    pub pad: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, i64, M) -> c_int,
    pub dilate: unsafe extern "C" fn(G, P, i64, i64, i64, i64, M) -> c_int,
    pub act: unsafe extern "C" fn(G, M, i64, i64, c_int, f32, *const f32, i64) -> c_int,
    pub pixel_shuffle: unsafe extern "C" fn(G, P, i64, i64, i64, i64, M) -> c_int,
    pub resize: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, c_int, M) -> c_int,
    pub add: unsafe extern "C" fn(G, P, P, i64, c_int, f32, f32, M) -> c_int,
    pub copy_channels: unsafe extern "C" fn(G, P, i64, i64, M, i64, i64) -> c_int,
    pub crop: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, M) -> c_int,
    pub mean: unsafe extern "C" fn(G, P, i64, i64, *mut f32) -> c_int,
    pub dense: unsafe extern "C" fn(G, *const f32, i64, *const f32, *const f32, i64, c_int, f32, *mut f32) -> c_int,
    pub scale: unsafe extern "C" fn(G, M, i64, i64, *const f32) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (the image library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in up.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            pre: sym!("ns_image_up_pre"),
            pre_alpha: sym!("ns_image_up_pre_alpha"),
            post: sym!("ns_image_up_post"),
            pad: sym!("ns_image_up_pad"),
            dilate: sym!("ns_image_up_dilate"),
            act: sym!("ns_image_up_act"),
            pixel_shuffle: sym!("ns_image_up_pixel_shuffle"),
            resize: sym!("ns_image_up_resize"),
            add: sym!("ns_image_up_add"),
            copy_channels: sym!("ns_image_up_copy_channels"),
            crop: sym!("ns_image_up_crop"),
            mean: sym!("ns_image_up_mean"),
            dense: sym!("ns_image_up_dense"),
            scale: sym!("ns_image_up_scale"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// The kernel's answer as a Result (the library's last error on -1)
pub fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("{what}: {why}")))
}

/// `g`'s handle for the calls
pub fn g(gpu: &Gpu) -> G {
    gpu.raw()
}

pub const SILO: &str = "upscale";

/// A silo convolution: x, H, W, Ci, x's pixel stride, w, Co, k, stride, pad top / left / bottom / right, bias, act, a,
/// slopes, n, residual (or null) and its two weights, out, out's pixel stride
pub type SiloConv = unsafe extern "C" fn(G, P, i64, i64, i64, i64, P, i64, i64, i64, i64, i64, i64, i64, *const f32, c_int, f32, *const f32, i64, P, f32, f32,
                                         M, i64) -> c_int;

/// The silo's kernels (kernels/image/upscale/silo/silo.h)
pub struct Silo {
    pub supported: unsafe extern "C" fn(G) -> c_int,
    /// a convolution with its zero padding and activation fused (act 3: PReLU, slopes given)
    pub conv: SiloConv,
    /// a channel slice copied into a wider buffer, 8 halves a work-item (src, P, Cs, dst, Cd, off)
    pub copy_channels: unsafe extern "C" fn(G, P, i64, i64, M, i64, i64) -> c_int,
    /// the nearest upsampling by a whole factor (x, H, W, C, r, out)
    pub upsample: unsafe extern "C" fn(G, P, i64, i64, i64, i64, M) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo, when built and this GPU runs it (None: NS_SILO=0, or no file - the shared convolutions throughout)
pub fn silo(gpu: &Gpu) -> Result<Option<&'static Silo>> {
    static S: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    let s = S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
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
        Ok(Some(Silo { supported: sym!("ns_image_up_silo_supported"), conv: sym!("ns_image_up_silo_conv"),
                        copy_channels: sym!("ns_image_up_silo_copy_channels"), upsample: sym!("ns_image_up_silo_upsample") }))
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))?;
    // SAFETY: the silo's entry point, on this GPU's context.
    Ok(s.as_ref().filter(|sl| unsafe { (sl.supported)(gpu.raw()) } == 1))
}
