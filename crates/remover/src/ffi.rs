//! The shared vision kernels (`kernels/diffusion/vision.h`, in every kind's library but the llm one), bound by name.
//! The pointers are device memory the host never dereferences; each call's doc in vision.h says the sizes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{Error, Gpu, Result};

type G = *mut c_void;
type P = *const f32;
type M = *mut f32;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub im2col: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, M) -> c_int,
    pub deform_im2col: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, P, i64, i64, M) -> c_int,
    pub resize_ac: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, M, i64) -> c_int,
    pub tiles: unsafe extern "C" fn(G, P, i64, i64, i64, i64, M) -> c_int,
    pub window_gather: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, M) -> c_int,
    pub window_scatter_add: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, M) -> c_int,
    pub window_attention: unsafe extern "C" fn(G, P, i64, i64, i64, i64, P, i64, f32, M) -> c_int,
    pub patch_merge: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    pub relu: unsafe extern "C" fn(G, M, i64) -> c_int,
    pub mul_sigmoid: unsafe extern "C" fn(G, M, i64, i64, P) -> c_int,
    pub copy_cols: unsafe extern "C" fn(G, P, i64, i64, i64, M, i64) -> c_int,
    pub col_mean: unsafe extern "C" fn(G, P, i64, i64, M) -> c_int,
    pub broadcast_row: unsafe extern "C" fn(G, P, i64, i64, M, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, M, P, i64) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (a library with kernels/diffusion/vision.cpp: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in vision.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            im2col: sym!("ns_vision_im2col"),
            deform_im2col: sym!("ns_vision_deform_im2col"),
            resize_ac: sym!("ns_vision_resize_ac"),
            tiles: sym!("ns_vision_tiles"),
            window_gather: sym!("ns_vision_window_gather"),
            window_scatter_add: sym!("ns_vision_window_scatter_add"),
            window_attention: sym!("ns_vision_window_attention"),
            patch_merge: sym!("ns_vision_patch_merge"),
            relu: sym!("ns_vision_relu"),
            mul_sigmoid: sym!("ns_vision_mul_sigmoid"),
            copy_cols: sym!("ns_vision_copy_cols"),
            col_mean: sym!("ns_vision_col_mean"),
            broadcast_row: sym!("ns_vision_broadcast_row"),
            add: sym!("ns_vision_add"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("{what}: {why}")))
}

/// The kernels on one GPU: checked calls by name
pub struct Vision {
    pub k: &'static Api,
    g: G,
}

// SAFETY: the GPU handle is a queue; calls go through &self from one engine thread at a time.
unsafe impl Send for Vision {}
unsafe impl Sync for Vision {}

impl Vision {
    pub fn new(gpu: &Gpu) -> Result<Vision> {
        Ok(Vision { k: api()?, g: gpu.raw() })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn im2col(&self, x: P, hwc: (usize, usize, usize), k: (usize, usize), s: usize, p: usize, d: usize, out_hw: (usize, usize), row0: usize, rows: usize,
                  out: M) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.im2col)(self.g, x, hwc.0 as i64, hwc.1 as i64, hwc.2 as i64, k.0 as i64, k.1 as i64, s as i64, p as i64, d as i64, out_hw.0 as i64,
                                        out_hw.1 as i64, row0 as i64, rows as i64, out) }, "im2col")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn deform_im2col(&self, x: P, hwc: (usize, usize, usize), k: usize, p: usize, om: P, row0: usize, rows: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.deform_im2col)(self.g, x, hwc.0 as i64, hwc.1 as i64, hwc.2 as i64, k as i64, p as i64, om, row0 as i64, rows as i64, out) },
              "deformable im2col")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn resize_ac(&self, x: P, hi: usize, wi: usize, c: usize, ho: usize, wo: usize, out: M, stride: usize) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.resize_ac)(self.g, x, hi as i64, wi as i64, c as i64, ho as i64, wo as i64, out, stride as i64) }, "bilinear resize")
    }
    pub fn tiles(&self, x: P, h: usize, w: usize, c: usize, g: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.tiles)(self.g, x, h as i64, w as i64, c as i64, g as i64, out) }, "tiles")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn window_gather(&self, x: P, h: usize, w: usize, c: usize, ws: usize, shift: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.window_gather)(self.g, x, h as i64, w as i64, c as i64, ws as i64, shift as i64, out) }, "window gather")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn window_scatter_add(&self, win: P, h: usize, w: usize, c: usize, ws: usize, shift: usize, x: M) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.window_scatter_add)(self.g, win, h as i64, w as i64, c as i64, ws as i64, shift as i64, x) }, "window scatter")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn window_attention(&self, qkv: P, nwh: usize, nww: usize, ws: usize, heads: usize, bias: P, shift: usize, scale: f32, out: M) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.window_attention)(self.g, qkv, nwh as i64, nww as i64, ws as i64, heads as i64, bias, shift as i64, scale, out) },
              "window attention")
    }
    pub fn patch_merge(&self, x: P, h: usize, w: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.patch_merge)(self.g, x, h as i64, w as i64, c as i64, out) }, "patch merge")
    }
    pub fn relu(&self, x: M, n: usize) -> Result<()> {
        // SAFETY: n floats.
        check(unsafe { (self.k.relu)(self.g, x, n as i64) }, "relu")
    }
    pub fn mul_sigmoid(&self, x: M, m: usize, c: usize, logit: P) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.mul_sigmoid)(self.g, x, m as i64, c as i64, logit) }, "sigmoid gate")
    }
    pub fn copy_cols(&self, src: P, m: usize, c: usize, ss: usize, dst: M, ds: usize) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.copy_cols)(self.g, src, m as i64, c as i64, ss as i64, dst, ds as i64) }, "column copy")
    }
    pub fn col_mean(&self, x: P, m: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.col_mean)(self.g, x, m as i64, c as i64, out) }, "column mean")
    }
    pub fn broadcast_row(&self, v: P, m: usize, c: usize, dst: M, ds: usize) -> Result<()> {
        // SAFETY: sizes as named (vision.h).
        check(unsafe { (self.k.broadcast_row)(self.g, v, m as i64, c as i64, dst, ds as i64) }, "row broadcast")
    }
    pub fn add(&self, x: M, y: P, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        check(unsafe { (self.k.add)(self.g, x, y, n as i64) }, "add")
    }
}
