//! This engine's own C ABI (`kernels/model3d/trellis2/t2.h`), bound from the 3D kernel library by name, and the calls
//! on one GPU (`K`). The pointers are device memory the host never dereferences; t2.h says each call's sizes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

type G = *mut c_void;
type P = *const c_void;
type M = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    linear_f32: unsafe extern "C" fn(G, P, i64, i64, P, i64, *const f32, *mut f32) -> c_int,
    ln_mod: unsafe extern "C" fn(G, P, c_int, i64, i64, f32, *const f32, *const f32, *const f32, *const f32, M, c_int) -> c_int,
    qk_norm_rope: unsafe extern "C" fn(G, M, i64, i64, i64, i64, *const f32, *const f32) -> c_int,
    gate_add: unsafe extern "C" fn(G, *mut f32, i64, i64, P, c_int, *const f32) -> c_int,
    convert: unsafe extern "C" fn(G, P, c_int, i64, M, c_int) -> c_int,
    silu: unsafe extern "C" fn(G, M, c_int, i64) -> c_int,
    add: unsafe extern "C" fn(G, M, c_int, P, c_int, i64) -> c_int,
    gather27: unsafe extern "C" fn(G, P, i64, *const i32, i64, i64, M) -> c_int,
    gather_rows: unsafe extern "C" fn(G, P, c_int, i64, *const i32, i64, M) -> c_int,
    c2s_skip: unsafe extern "C" fn(G, *const f32, i64, *const i32, *const i32, i64, i64, i64, *mut f32) -> c_int,
    pixel_shuffle3d: unsafe extern "C" fn(G, P, i64, i64, M) -> c_int,
    grid_sample2d: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const f32, i64, *mut f32, i64) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// The table, bound once (an error when the library lacks a symbol: an older build)
pub fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (rebuild it: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in t2.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            linear_f32: sym!("ns_model3d_trellis2_linear_f32"),
            ln_mod: sym!("ns_model3d_trellis2_ln_mod"),
            qk_norm_rope: sym!("ns_model3d_trellis2_qk_norm_rope"),
            gate_add: sym!("ns_model3d_trellis2_gate_add"),
            convert: sym!("ns_model3d_trellis2_convert"),
            silu: sym!("ns_model3d_trellis2_silu"),
            add: sym!("ns_model3d_trellis2_add"),
            gather27: sym!("ns_model3d_trellis2_gather27"),
            gather_rows: sym!("ns_model3d_trellis2_gather_rows"),
            c2s_skip: sym!("ns_model3d_trellis2_c2s_skip"),
            pixel_shuffle3d: sym!("ns_model3d_trellis2_pixel_shuffle3d"),
            grid_sample2d: sym!("ns_model3d_trellis2_grid_sample2d"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("{what}: {why}")))
}

fn d(dt: Dt) -> c_int {
    dt as c_int
}

/// The kernels on one GPU
pub struct K {
    k: &'static Api,
    g: G,
}

// SAFETY: the GPU handle is a queue; the engine calls through &self from one thread at a time.
unsafe impl Send for K {}
unsafe impl Sync for K {}

impl K {
    pub fn new(gpu: &Gpu) -> Result<K> {
        Ok(K { k: api()?, g: gpu.raw() })
    }

    /// out [m, n] float32 = x [m, k] half . w [n, k] half ^T + bias
    #[allow(clippy::too_many_arguments)]
    pub fn linear_f32(&self, x: P, m: usize, k: usize, w: P, n: usize, bias: *const f32, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (t2.h).
        check(unsafe { (self.k.linear_f32)(self.g, x, m as i64, k as i64, w, n as i64, bias, out) }, "linear (float32 out)")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn ln_mod(&self, x: P, x_dt: Dt, m: usize, c: usize, eps: f32, w: *const f32, b: *const f32, scale: *const f32, shift: *const f32, out: M,
                  out_dt: Dt) -> Result<()> {
        // SAFETY: sizes as named (t2.h).
        check(unsafe { (self.k.ln_mod)(self.g, x, d(x_dt), m as i64, c as i64, eps, w, b, scale, shift, out, d(out_dt)) }, "layer norm")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(&self, x: M, m: usize, h: usize, dd: usize, stride: usize, gamma: *const f32, cs: *const f32) -> Result<()> {
        // SAFETY: sizes as named (t2.h).
        check(unsafe { (self.k.qk_norm_rope)(self.g, x, m as i64, h as i64, dd as i64, stride as i64, gamma, cs) }, "qk norm + rope")
    }
    pub fn gate_add(&self, x: *mut f32, m: usize, c: usize, h: P, h_dt: Dt, gate: *const f32) -> Result<()> {
        // SAFETY: sizes as named (t2.h).
        check(unsafe { (self.k.gate_add)(self.g, x, m as i64, c as i64, h, d(h_dt), gate) }, "gated add")
    }
    pub fn convert(&self, src: P, src_dt: Dt, n: usize, dst: M, dst_dt: Dt) -> Result<()> {
        // SAFETY: n values each.
        check(unsafe { (self.k.convert)(self.g, src, d(src_dt), n as i64, dst, d(dst_dt)) }, "convert")
    }
    pub fn silu(&self, x: M, dt: Dt, n: usize) -> Result<()> {
        // SAFETY: n values.
        check(unsafe { (self.k.silu)(self.g, x, d(dt), n as i64) }, "silu")
    }
    pub fn add(&self, x: M, x_dt: Dt, y: P, y_dt: Dt, n: usize) -> Result<()> {
        // SAFETY: n values each.
        check(unsafe { (self.k.add)(self.g, x, d(x_dt), y, d(y_dt), n as i64) }, "add")
    }
    pub fn gather27(&self, feats: P, c: usize, nmap: *const i32, row0: usize, rows: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (t2.h).
        check(unsafe { (self.k.gather27)(self.g, feats, c as i64, nmap, row0 as i64, rows as i64, out) }, "sparse gather")
    }
    pub fn gather_rows(&self, src: P, dt: Dt, c: usize, idx: *const i32, n: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (t2.h).
        check(unsafe { (self.k.gather_rows)(self.g, src, d(dt), c as i64, idx, n as i64, out) }, "row gather")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn c2s_skip(&self, x: *const f32, cin: usize, parent: *const i32, sub: *const i32, n: usize, cout: usize, r: usize, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (t2.h).
        check(unsafe { (self.k.c2s_skip)(self.g, x, cin as i64, parent, sub, n as i64, cout as i64, r as i64, out) }, "subdivision skip")
    }
    pub fn pixel_shuffle3d(&self, x: P, dd: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (t2.h).
        check(unsafe { (self.k.pixel_shuffle3d)(self.g, x, dd as i64, c as i64, out) }, "pixel shuffle")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn grid_sample2d(&self, feat: *const f32, hf: usize, wf: usize, c: usize, uv: *const f32, n: usize, out: *mut f32, stride: usize) -> Result<()> {
        // SAFETY: sizes as named (t2.h).
        check(unsafe { (self.k.grid_sample2d)(self.g, feat, hf as i64, wf as i64, c as i64, uv, n as i64, out, stride as i64) }, "grid sample")
    }
}
