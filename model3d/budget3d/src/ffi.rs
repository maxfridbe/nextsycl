//! This engine's own C ABI (`kernels/model3d/budget3d/b3d.h`), bound from the 3D kernel library by name, and the calls
//! on one GPU (`K`). The pointers are device memory the host never dereferences; the header says each call's sizes.
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
    ln_mod: unsafe extern "C" fn(G, P, c_int, i64, i64, f32, *const f32, *const f32, *const f32, *const f32, i64, i64, M, c_int) -> c_int,
    gate_add: unsafe extern "C" fn(G, *mut f32, i64, i64, P, c_int, *const f32, i64, i64) -> c_int,
    rope_norm: unsafe extern "C" fn(G, M, i64, i64, i64, i64, *const f32, *const f32, i64, i64, *const f32) -> c_int,
    small_attn: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, *mut f32) -> c_int,
    gather_add: unsafe extern "C" fn(G, *const f32, i64, *const i32, i64, *const f32, *const i32, *mut f32) -> c_int,
    slot_add: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const f32, *mut f32) -> c_int,
    mix: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const f32, *const f32, *mut f32) -> c_int,
    group_mean: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32) -> c_int,
    refine_init: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const f32, *const f32, *const f32, *mut f32) -> c_int,
    convert: unsafe extern "C" fn(G, P, c_int, i64, M, c_int) -> c_int,
    silu: unsafe extern "C" fn(G, M, c_int, i64) -> c_int,
    add: unsafe extern "C" fn(G, M, c_int, P, c_int, i64) -> c_int,
    group_norm: unsafe extern "C" fn(G, P, i64, i64, i64, f32, *const f32, *const f32, c_int, M, *mut f32) -> c_int,
    pad_br: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    softmax_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M) -> c_int,
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
                // SAFETY: the symbol is declared in b3d.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            linear_f32: sym!("ns_model3d_budget3d_linear_f32"),
            ln_mod: sym!("ns_model3d_budget3d_ln_mod"),
            gate_add: sym!("ns_model3d_budget3d_gate_add"),
            rope_norm: sym!("ns_model3d_budget3d_rope_norm"),
            small_attn: sym!("ns_model3d_budget3d_small_attn"),
            gather_add: sym!("ns_model3d_budget3d_gather_add"),
            slot_add: sym!("ns_model3d_budget3d_slot_add"),
            mix: sym!("ns_model3d_budget3d_mix"),
            group_mean: sym!("ns_model3d_budget3d_group_mean"),
            refine_init: sym!("ns_model3d_budget3d_refine_init"),
            convert: sym!("ns_model3d_budget3d_convert"),
            silu: sym!("ns_model3d_budget3d_silu"),
            add: sym!("ns_model3d_budget3d_add"),
            group_norm: sym!("ns_model3d_budget3d_group_norm"),
            pad_br: sym!("ns_model3d_budget3d_pad_br"),
            softmax_rows: sym!("ns_model3d_budget3d_softmax_rows"),
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

/// A modulation table: row r of the call reads `p + (r / group) * stride` (group 0: `p` for every row)
#[derive(Clone, Copy)]
pub struct Rows {
    pub scale: *const f32,
    pub shift: *const f32,
    pub stride: usize,
    pub group: usize,
}

impl Rows {
    /// No modulation
    pub fn none() -> Rows {
        Rows { scale: std::ptr::null(), shift: std::ptr::null(), stride: 0, group: 0 }
    }
    /// One (scale, shift) for every row
    pub fn one(scale: *const f32, shift: *const f32) -> Rows {
        Rows { scale, shift, stride: 0, group: 0 }
    }
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
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.linear_f32)(self.g, x, m as i64, k as i64, w, n as i64, bias, out) }, "linear (float32 out)")
    }
    /// Layer norm (affine `w`, `b` when given), then the modulation `md`
    #[allow(clippy::too_many_arguments)]
    pub fn ln_mod(&self, x: P, x_dt: Dt, m: usize, c: usize, eps: f32, w: *const f32, b: *const f32, md: Rows, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.ln_mod)(self.g, x, d(x_dt), m as i64, c as i64, eps, w, b, md.scale, md.shift, md.stride as i64, md.group as i64, out, d(out_dt)) },
              "layer norm")
    }
    /// x [m, c] += h * gate (row r's gate at (r / group) * stride; group 0: one gate row; gate null: a plain add)
    #[allow(clippy::too_many_arguments)]
    pub fn gate_add(&self, x: *mut f32, m: usize, c: usize, h: P, h_dt: Dt, gate: *const f32, stride: usize, group: usize) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.gate_add)(self.g, x, m as i64, c as i64, h, d(h_dt), gate, stride as i64, group as i64) }, "gated add")
    }
    /// RePo rotation (delta null: none) then the per-head RMS norm, in place on half rows
    #[allow(clippy::too_many_arguments)]
    pub fn rope_norm(&self, x: M, m: usize, h: usize, dd: usize, stride: usize, delta: *const f32, freqs: *const f32, n0: usize, n1: usize,
                     gamma: *const f32) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.rope_norm)(self.g, x, m as i64, h as i64, dd as i64, stride as i64, delta, freqs, n0 as i64, n1 as i64, gamma) },
              "rotary positions + RMS norm")
    }
    pub fn small_attn(&self, qkv: *const f32, b: usize, s: usize, h: usize, dd: usize, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.small_attn)(self.g, qkv, b as i64, s as i64, h as i64, dd as i64, out) }, "refiner attention")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gather_add(&self, src: *const f32, c: usize, idx: *const i32, n: usize, add: *const f32, aidx: *const i32, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.gather_add)(self.g, src, c as i64, idx, n as i64, add, aidx, out) }, "row gather")
    }
    pub fn slot_add(&self, x: *const f32, r: usize, s: usize, c: usize, pos: *const f32, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.slot_add)(self.g, x, r as i64, s as i64, c as i64, pos, out) }, "slot add")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn mix(&self, h: *const f32, r: usize, s: usize, c: usize, w: *const f32, b: *const f32, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.mix)(self.g, h, r as i64, s as i64, c as i64, w, b, out) }, "scale mixer")
    }
    pub fn group_mean(&self, x: *const f32, r: usize, s: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.group_mean)(self.g, x, r as i64, s as i64, c as i64, out) }, "group mean")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn refine_init(&self, e: *const f32, r: usize, s: usize, c: usize, slot: *const f32, zt: *const f32, carry: *const f32, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.refine_init)(self.g, e, r as i64, s as i64, c as i64, slot, zt, carry, out) }, "refiner input")
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
    /// Group norm (+ SiLU) of a channels-last half map; `scratch` float32 [groups * 66]
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm(&self, x: P, px: usize, c: usize, groups: usize, eps: f32, gamma: *const f32, beta: *const f32, silu: bool, out: M,
                      scratch: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.group_norm)(self.g, x, px as i64, c as i64, groups as i64, eps, gamma, beta, silu as c_int, out, scratch) }, "group norm")
    }
    pub fn pad_br(&self, x: P, h: usize, w: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.pad_br)(self.g, x, h as i64, w as i64, c as i64, out) }, "pad")
    }
    pub fn softmax_rows(&self, x: *const f32, m: usize, n: usize, scale: f32, out: M) -> Result<()> {
        // SAFETY: sizes as named (b3d.h).
        check(unsafe { (self.k.softmax_rows)(self.g, x, m as i64, n as i64, scale, out) }, "softmax")
    }
}
