//! Raw bindings to `libnextsycl.so`: one function pointer per declaration of `kernels/ns/ns.h`.
//!
//! The library is opened at run time (`dlopen`), not linked: the Rust side builds anywhere without the SYCL
//! toolchain, and the kernels rebuild without relinking the runtime. Nothing here is safe to call directly;
//! `ns-core` wraps it.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

/// An opened GPU (`ns_gpu*`).
pub type Gpu = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub last_error: unsafe extern "C" fn() -> *const c_char,
    pub version: unsafe extern "C" fn() -> *const c_char,
    pub gpu_count: unsafe extern "C" fn() -> c_int,
    pub gpu_name: unsafe extern "C" fn(c_int, *mut c_char, usize) -> c_int,
    pub gpu_open: unsafe extern "C" fn(c_int, *mut Gpu) -> c_int,
    pub gpu_close: unsafe extern "C" fn(Gpu),
    pub gpu_queue: unsafe extern "C" fn(Gpu) -> *mut c_void,
    pub gpu_memory: unsafe extern "C" fn(Gpu, *mut u64, *mut i64) -> c_int,
    pub alloc: unsafe extern "C" fn(Gpu, usize, *mut *mut c_void) -> c_int,
    pub free: unsafe extern "C" fn(Gpu, *mut c_void) -> c_int,
    pub alloc_host: unsafe extern "C" fn(Gpu, usize, *mut *mut c_void) -> c_int,
    pub free_host: unsafe extern "C" fn(Gpu, *mut c_void) -> c_int,
    pub copy_to: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize) -> c_int,
    pub copy_from: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize) -> c_int,
    pub copy_dev: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize) -> c_int,
    pub fill: unsafe extern "C" fn(Gpu, *mut c_void, u8, usize) -> c_int,
    pub sync: unsafe extern "C" fn(Gpu) -> c_int,
    pub copy_peer: unsafe extern "C" fn(Gpu, *mut c_void, Gpu, *const c_void, usize, *mut c_void) -> c_int,
    // the bring-up kernels (glm.cpp)
    pub dequant: unsafe extern "C" fn(Gpu, c_int, *const c_void, usize, *mut f32) -> c_int,
    pub gemm: unsafe extern "C" fn(Gpu, i64, i64, i64, *const f32, i64, *const f32, *mut f32, i64, c_int) -> c_int,
    pub rms_norm: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub layer_norm: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub hc_pre: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32, i64, i64, f32, c_int) -> c_int,
    pub hc_post: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *mut f32, i64, i64) -> c_int,
    pub hc_mean: unsafe extern "C" fn(Gpu, *const f32, *mut f32, i64, i64) -> c_int,
    pub conv_silu: unsafe extern "C" fn(Gpu, *const f32, *mut f32, *const f32, *mut f32, i64, i64, c_int) -> c_int,
    pub l2_norm: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, f32) -> c_int,
    pub kda_gate: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const f32, i64, i64, i64, f32) -> c_int,
    pub sigmoid: unsafe extern "C" fn(Gpu, *mut f32, i64) -> c_int,
    pub kda_scan: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, i64, i64, i64) -> c_int,
    pub kda_out: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *mut f32, i64, i64, i64, f32) -> c_int,
    pub swiglu_clamp: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, f32) -> c_int,
    pub mla_attend: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, i64, i64, i64, f32) -> c_int,
    pub scatter_add: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const i32, *const f32, i64, i64) -> c_int,
    pub gather: unsafe extern "C" fn(Gpu, *const f32, *const i32, *mut f32, i64, i64) -> c_int,
    pub add: unsafe extern "C" fn(Gpu, *mut f32, *const f32, i64) -> c_int,
    // decode-width products (mmvq.cpp)
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub quantize_q8_1: unsafe extern "C" fn(Gpu, *const f32, *mut c_void, i64, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(Gpu, c_int, *const c_void, *const c_void, *mut f32, i64, i64, i64) -> c_int,
}

// SAFETY: plain function pointers into a library that stays loaded for the process's life.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn dl_error() -> String {
    // SAFETY: dlerror returns null or a C string valid until the next dl* call on this thread.
    let p = unsafe { dlerror() };
    if p.is_null() {
        "unknown dlopen error".into()
    } else {
        // SAFETY: non-null, NUL-terminated (above).
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

impl Api {
    /// `NEXTSYCL_LIB`, else `libnextsycl.so` beside the executable, else `dist/` of the working directory.
    pub fn load() -> Result<Api, String> {
        let p = std::env::var_os("NEXTSYCL_LIB").map(PathBuf::from).or_else(|| {
            let exe = std::env::current_exe().ok()?;
            let beside = exe.parent()?.join("libnextsycl.so");
            beside.exists().then_some(beside)
        });
        Api::load_from(&p.unwrap_or_else(|| PathBuf::from("dist/libnextsycl.so")))
    }

    #[allow(clippy::missing_transmute_annotations)] // the target type is each field's, from ns.h
    pub fn load_from(path: &Path) -> Result<Api, String> {
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: dlopen with a valid C string; the handle is checked.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if h.is_null() {
            return Err(format!("{}: {}", path.display(), dl_error()));
        }
        macro_rules! sym {
            ($n:literal) => {{
                let n = CString::new($n).unwrap();
                // SAFETY: a symbol lookup in the handle above; the pointer is checked and has ns.h's signature.
                let p = unsafe { dlsym(h, n.as_ptr()) };
                if p.is_null() {
                    return Err(format!("{}: no symbol {}", path.display(), $n));
                }
                // SAFETY: the declaration in ns.h is the field's type.
                unsafe { std::mem::transmute::<*mut c_void, _>(p) }
            }};
        }
        Ok(Api {
            last_error: sym!("ns_last_error"),
            version: sym!("ns_version"),
            gpu_count: sym!("ns_gpu_count"),
            gpu_name: sym!("ns_gpu_name"),
            gpu_open: sym!("ns_gpu_open"),
            gpu_close: sym!("ns_gpu_close"),
            gpu_queue: sym!("ns_gpu_queue"),
            gpu_memory: sym!("ns_gpu_memory"),
            alloc: sym!("ns_alloc"),
            free: sym!("ns_free"),
            alloc_host: sym!("ns_alloc_host"),
            free_host: sym!("ns_free_host"),
            copy_to: sym!("ns_copy_to"),
            copy_from: sym!("ns_copy_from"),
            copy_dev: sym!("ns_copy_dev"),
            fill: sym!("ns_fill"),
            sync: sym!("ns_sync"),
            copy_peer: sym!("ns_copy_peer"),
            dequant: sym!("ns_dequant"),
            gemm: sym!("ns_gemm"),
            rms_norm: sym!("ns_rms_norm"),
            layer_norm: sym!("ns_layer_norm"),
            hc_pre: sym!("ns_hc_pre"),
            hc_post: sym!("ns_hc_post"),
            hc_mean: sym!("ns_hc_mean"),
            conv_silu: sym!("ns_conv_silu"),
            l2_norm: sym!("ns_l2_norm"),
            kda_gate: sym!("ns_kda_gate"),
            sigmoid: sym!("ns_sigmoid"),
            kda_scan: sym!("ns_kda_scan"),
            kda_out: sym!("ns_kda_out"),
            swiglu_clamp: sym!("ns_swiglu_clamp"),
            mla_attend: sym!("ns_mla_attend"),
            scatter_add: sym!("ns_scatter_add"),
            gather: sym!("ns_gather"),
            add: sym!("ns_add"),
            q8_1_bytes: sym!("ns_q8_1_bytes"),
            mmvq_supported: sym!("ns_mmvq_supported"),
            quantize_q8_1: sym!("ns_quantize_q8_1"),
            mmvq: sym!("ns_mmvq"),
        })
    }

    /// The reason of the last failed call on this thread.
    pub fn error(&self) -> String {
        // SAFETY: ns_last_error returns a C string owned by the library, valid until the next call on this thread.
        unsafe { CStr::from_ptr((self.last_error)()) }.to_string_lossy().into_owned()
    }
}
