//! Raw bindings to the diffusion kernels (`kernels/diffusion/nsd.h`, H3's `h3sycl.h` under the `nsd_` prefix) in the
//! video kind's library (`libnextsycl-video.so`): one function pointer per declaration, resolved by name at run
//! time, so the Rust side builds anywhere and the kernels rebuild without relinking. H3's binding (`h3-sys`) as it
//! was, with the library nextsycl-core opens for the video kind. Nothing here is safe to call directly; the
//! engine's `device` and `ops` wrap it.

use std::ffi::{c_char, c_int, c_void, CStr};

/// Element types of floating-point tensors (`H3S_F32` ...).
pub const F32: c_int = 0;
pub const F16: c_int = 1;
pub const BF16: c_int = 2;


/// The functions of `h3sycl.h`, resolved once.
#[allow(non_snake_case)]
pub struct Api {
    pub last_error: unsafe extern "C" fn() -> *const c_char,
    pub open: unsafe extern "C" fn() -> *mut c_void,
    pub gpu_count: unsafe extern "C" fn() -> c_int,
    pub gpu_info: unsafe extern "C" fn(c_int, *mut c_char, c_int, *mut u64, *mut c_char, c_int) -> c_int,
    pub open_gpu: unsafe extern "C" fn(c_int) -> *mut c_void,
    pub destroy: unsafe extern "C" fn(*mut c_void),
    pub device_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    pub alloc: unsafe extern "C" fn(*mut c_void, u64) -> *mut c_void,
    pub free: unsafe extern "C" fn(*mut c_void, *mut c_void),
    pub mem_used: unsafe extern "C" fn(*mut c_void) -> u64,
    pub mem_cap: unsafe extern "C" fn(*mut c_void) -> u64,
    pub mem_free: unsafe extern "C" fn(*mut c_void) -> u64,
    pub write: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, u64) -> c_int,
    pub read: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, u64) -> c_int,
    pub wait: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub copy: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, u64) -> c_int,
    pub alloc_host: unsafe extern "C" fn(*mut c_void, u64) -> *mut c_void,
    pub free_host: unsafe extern "C" fn(*mut c_void, *mut c_void),
    pub upload: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, u64) -> c_int,
    pub upload_wait: unsafe extern "C" fn(*mut c_void) -> c_int,
    #[allow(clippy::type_complexity)]
    pub int8_linear: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *const c_void,  // x
        c_int,          // x_dt
        i64,            // M
        i64,            // K
        *const i8,      // w
        i64,            // N
        *const f32,     // wscale
        i64,            // n_wscale
        *const f32,     // bias
        *mut c_void,    // out
        c_int,          // out_dt
        c_int,          // group
    ) -> c_int,
    #[allow(clippy::type_complexity)]
    pub linear_acc: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, *const c_void, i64, *const f32, *mut c_void) -> c_int,
    pub scale_rows: unsafe extern "C" fn(*mut c_void, *mut c_void, c_int, i64, i64, *const f32) -> c_int,
    pub linear: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *const c_void,  // x
        c_int,          // dt
        i64,            // M
        i64,            // K
        *const c_void,  // w
        i64,            // N
        *const f32,     // bias
        *mut c_void,    // out
        c_int,          // out_dt
    ) -> c_int,
    #[allow(clippy::type_complexity)]
    pub dequant: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, *mut c_void, c_int) -> c_int,
    pub attention_causal: unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *const c_void, c_int, i64, i64, i64, i64, i64, i64, *mut c_void) -> c_int,
    pub conv3d: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, i64, i64, *const c_void, i64, i64, *const f32, *mut c_void) -> c_int,
    pub conv2d: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, i64, i64, *const c_void, i64, i64, *const f32, *mut c_void) -> c_int,
    pub prelu: unsafe extern "C" fn(*mut c_void, *mut c_void, c_int, i64, i64, *const f32) -> c_int,
    pub pixel_shuffle_add: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, i64, i64, i64, *const c_void, *mut c_void) -> c_int,
    pub resize_area: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub group_norm_silu: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, i64, i64, *const f32, *const f32, f32, *const f32, *const f32, *mut c_void) -> c_int,
    pub pad3d: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, i64, i64, i64, i64, i64, i64, i64, *mut c_void) -> c_int,
    pub conv3d_ex: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, i64, i64, *const c_void, i64, i64, i64, i64, i64, i64, i64, *const f32, *mut c_void) -> c_int,
    pub temporal_dwconv: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, i64, *const f32, i64, *const f32, *mut c_void) -> c_int,
    pub trilinear: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, i64, i64, i64, i64, i64, *mut c_void) -> c_int,
    pub conv1d: unsafe extern "C" fn(*mut c_void, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, i64, *mut f32, i64) -> c_int,
    pub conv_transpose1d: unsafe extern "C" fn(*mut c_void, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, *mut f32, i64) -> c_int,
    pub aa_snake: unsafe extern "C" fn(*mut c_void, *const f32, i64, i64, i64, *const f32, *const f32, *const f32, *const f32, *mut f32) -> c_int,
    pub scale: unsafe extern "C" fn(*mut c_void, *mut f32, i64, f32) -> c_int,
    pub snake: unsafe extern "C" fn(*mut c_void, *const f32, i64, i64, i64, *const f32, *mut f32) -> c_int,
    pub layer_norm: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, *const f32, *const f32, f32, *mut c_void, c_int) -> c_int,
    pub rms_norm_mod: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *const c_void,  // x
        c_int,          // x_dt
        i64,            // M
        i64,            // C
        *const f32,     // weight
        f32,            // eps
        *const i32,     // rows
        *const f32,     // scale
        *const f32,     // shift
        *mut c_void,    // out
        c_int,          // out_dt
    ) -> c_int,
    #[allow(clippy::type_complexity)]
    pub rms_rope: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *mut c_void,    // x
        c_int,          // x_dt
        i64,            // M
        i64,            // H
        i64,            // D
        i64,            // stride
        *const f32,     // weight
        f32,            // eps
        *const f32,     // cs
        c_int,          // rot_dim
    ) -> c_int,
    pub swiglu: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, *mut c_void, c_int) -> c_int,
    #[allow(clippy::type_complexity)]
    pub gate_add: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *mut c_void,    // x
        c_int,          // x_dt
        i64,            // M
        i64,            // C
        *const c_void,  // other
        c_int,          // other_dt
        *const i32,     // rows
        *const f32,     // gate
    ) -> c_int,
    #[allow(clippy::type_complexity)]
    pub attention: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *const c_void,  // q
        *const c_void,  // k
        *const c_void,  // v
        c_int,          // dt
        i64,            // S
        i64,            // H
        i64,            // D
        i64,            // stride
        *mut c_void,    // out
        c_int,          // out_dt
    ) -> c_int,
    #[allow(clippy::type_complexity)]
    pub attention_batch: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *const c_void,  // q
        *const c_void,  // k
        *const c_void,  // v
        c_int,          // dt
        i64,            // B
        i64,            // S
        i64,            // H
        i64,            // D
        i64,            // stride
        *mut c_void,    // out
        c_int,          // out_dt
    ) -> c_int,
}


impl Api {
    /// Resolves every function in the video kind's kernel library (nextsycl-core opens it; it stays loaded as long
    /// as the process).
    pub fn load() -> Result<Api, String> {
        nextsycl_core::use_kind("video");
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        Self::resolve(&|name: &str| lib.symbol(name))
    }

    // each field's declared type is the annotation: the transmute target is inferred from it
    #[allow(clippy::missing_transmute_annotations)]
    fn resolve(find: &dyn Fn(&str) -> *mut c_void) -> Result<Api, String> {
        macro_rules! sym {
            ($name:literal) => {{
                let p = find($name);
                if p.is_null() {
                    return Err(format!("{} is not in the video kernel library (./build.sh kernels)", $name));
                }
                // SAFETY: the symbol's type is the one declared in nsd.h.
                unsafe { std::mem::transmute::<*mut c_void, _>(p) }
            }};
        }
        Ok(Api {
            last_error: sym!("nsd_last_error"),
            open: sym!("nsd_open"),
            gpu_count: sym!("nsd_gpu_count"),
            gpu_info: sym!("nsd_gpu_info"),
            open_gpu: sym!("nsd_open_gpu"),
            destroy: sym!("nsd_destroy"),
            device_name: sym!("nsd_device_name"),
            alloc: sym!("nsd_alloc"),
            free: sym!("nsd_free"),
            mem_used: sym!("nsd_mem_used"),
            mem_cap: sym!("nsd_mem_cap"),
            mem_free: sym!("nsd_mem_free"),
            write: sym!("nsd_write"),
            read: sym!("nsd_read"),
            wait: sym!("nsd_wait"),
            copy: sym!("nsd_copy"),
            alloc_host: sym!("nsd_alloc_host"),
            free_host: sym!("nsd_free_host"),
            upload: sym!("nsd_upload"),
            upload_wait: sym!("nsd_upload_wait"),
            int8_linear: sym!("nsd_int8_linear"),
            linear: sym!("nsd_linear"),
            linear_acc: sym!("nsd_linear_acc"),
            scale_rows: sym!("nsd_scale_rows"),
            dequant: sym!("nsd_dequant"),
            attention_causal: sym!("nsd_attention_causal"),
            conv3d: sym!("nsd_conv3d"),
            conv2d: sym!("nsd_conv2d"),
            prelu: sym!("nsd_prelu"),
            pixel_shuffle_add: sym!("nsd_pixel_shuffle_add"),
            resize_area: sym!("nsd_resize_area"),
            group_norm_silu: sym!("nsd_group_norm_silu"),
            pad3d: sym!("nsd_pad3d"),
            conv3d_ex: sym!("nsd_conv3d_ex"),
            temporal_dwconv: sym!("nsd_temporal_dwconv"),
            trilinear: sym!("nsd_trilinear"),
            conv1d: sym!("nsd_conv1d"),
            conv_transpose1d: sym!("nsd_conv_transpose1d"),
            aa_snake: sym!("nsd_aa_snake"),
            scale: sym!("nsd_scale"),
            snake: sym!("nsd_snake"),
            layer_norm: sym!("nsd_layer_norm"),
            rms_norm_mod: sym!("nsd_rms_norm_mod"),
            rms_rope: sym!("nsd_rms_rope"),
            swiglu: sym!("nsd_swiglu"),
            gate_add: sym!("nsd_gate_add"),
            attention: sym!("nsd_attention"),
            attention_batch: sym!("nsd_attention_batch"),
        })
    }

    /// The library's last error on this thread.
    pub fn error(&self) -> String {
        // SAFETY: nsd_last_error returns a NUL-terminated string that lives until the thread's next failing call.
        unsafe { CStr::from_ptr((self.last_error)()).to_string_lossy().into_owned() }
    }
}
