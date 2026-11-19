//! This engine's own C ABI (`kernels/model3d/hi3d/hi3d.h`), bound from the 3D kernel library by name, its silo's
//! (`kernels/model3d/hi3d/silo/h3s.h`, dist/silo/libnextsycl-hi3d.so when it is there and NS_SILO is not 0), and the
//! calls on one GPU (`K`: the silo's kernel where it covers a call, hi3d.h's otherwise). The pointers are device memory
//! the host never dereferences; the headers say each call's sizes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

type G = *mut c_void;
type P = *const c_void;
type M = *mut c_void;
type I = *const i32;

#[allow(clippy::type_complexity)]
pub struct Api {
    linear_f32: unsafe extern "C" fn(G, P, i64, i64, P, i64, *const f32, *mut f32) -> c_int,
    ln_mod: unsafe extern "C" fn(G, P, c_int, i64, i64, f32, *const f32, *const f32, *const f32, *const f32, M, c_int) -> c_int,
    rope_rms: unsafe extern "C" fn(G, M, i64, i64, i64, i64, *const f32, *const f32) -> c_int,
    gate_add: unsafe extern "C" fn(G, *mut f32, i64, i64, P, c_int, *const f32) -> c_int,
    add_row: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32) -> c_int,
    convert: unsafe extern "C" fn(G, P, c_int, i64, M, c_int) -> c_int,
    act: unsafe extern "C" fn(G, M, c_int, i64, c_int) -> c_int,
    add: unsafe extern "C" fn(G, M, c_int, P, c_int, i64) -> c_int,
    put_cols: unsafe extern "C" fn(G, P, c_int, i64, i64, M, c_int, i64, i64) -> c_int,
    gather27: unsafe extern "C" fn(G, P, i64, I, i64, i64, M) -> c_int,
    gather_rows: unsafe extern "C" fn(G, P, c_int, i64, I, i64, M) -> c_int,
    seg_mean: unsafe extern "C" fn(G, P, c_int, i64, I, I, i64, *mut f32) -> c_int,
    gn_sums: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, *mut f32) -> c_int,
    gn_swish: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *const f32, *const f32, *const f32, *const f32, M, c_int) -> c_int,
    win_attn: unsafe extern "C" fn(G, P, i64, i64, i64, i64, f32, M) -> c_int,
    child_nmap: unsafe extern "C" fn(G, I, I, I, I, i64, *mut i32) -> c_int,
    copy_heads: unsafe extern "C" fn(G, P, i64, i64, i64, i64, M, i64, i64, f32) -> c_int,
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
                // SAFETY: the symbol is declared in hi3d.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            linear_f32: sym!("ns_model3d_hi3d_linear_f32"),
            ln_mod: sym!("ns_model3d_hi3d_ln_mod"),
            rope_rms: sym!("ns_model3d_hi3d_rope_rms"),
            gate_add: sym!("ns_model3d_hi3d_gate_add"),
            add_row: sym!("ns_model3d_hi3d_add_row"),
            convert: sym!("ns_model3d_hi3d_convert"),
            act: sym!("ns_model3d_hi3d_act"),
            add: sym!("ns_model3d_hi3d_add"),
            put_cols: sym!("ns_model3d_hi3d_put_cols"),
            gather27: sym!("ns_model3d_hi3d_gather27"),
            gather_rows: sym!("ns_model3d_hi3d_gather_rows"),
            seg_mean: sym!("ns_model3d_hi3d_seg_mean"),
            gn_sums: sym!("ns_model3d_hi3d_gn_sums"),
            gn_swish: sym!("ns_model3d_hi3d_gn_swish"),
            win_attn: sym!("ns_model3d_hi3d_win_attn"),
            child_nmap: sym!("ns_model3d_hi3d_child_nmap"),
            copy_heads: sym!("ns_model3d_hi3d_copy_heads"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// The silo's kernels (h3s.h)
#[allow(clippy::type_complexity)]
pub struct Silo {
    pub path: std::path::PathBuf,
    linear_supported: unsafe extern "C" fn(c_int, c_int) -> c_int,
    linear: unsafe extern "C" fn(G, P, i64, i64, P, i64, *const f32, M, c_int, c_int, *const f32) -> c_int,
    ln_mod_supported: unsafe extern "C" fn(i64, c_int, c_int) -> c_int,
    ln_mod: unsafe extern "C" fn(G, P, c_int, i64, i64, f32, *const f32, *const f32, *const f32, *const f32, M, c_int) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo, opened once (None: there is none, or NS_SILO=0 - hi3d.h's kernels for everything)
pub fn silo() -> Result<Option<&'static Silo>> {
    static SILO: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    SILO.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(crate::ARCH).map_err(|e| e.0)? else { return Ok(None) };
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{}: no {} (rebuild it: ./build.sh kernels)", lib.path.display(), $name));
                }
                // SAFETY: the symbol is declared in h3s.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Some(Silo {
            path: lib.path.clone(),
            linear_supported: sym!("ns_hi3ds_linear_supported"),
            linear: sym!("ns_hi3ds_linear"),
            ln_mod_supported: sym!("ns_hi3ds_ln_mod_supported"),
            ln_mod: sym!("ns_hi3ds_ln_mod"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| Error(e.clone()))
}

/// What a product does to its result (h3s.h's epilogues)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Epi {
    /// out = x w^T + bias
    Plain,
    /// out = gelu_tanh(x w^T + bias)
    Gelu,
    /// out = gelu(x w^T + bias), erf
    GeluErf,
    /// out = silu(x w^T + bias)
    Silu,
    /// out (float32) += (x w^T + bias) * gate (null: a plain add)
    Resid(*const f32),
}

impl Epi {
    fn code(self) -> (c_int, *const f32) {
        match self {
            Epi::Plain => (0, std::ptr::null()),
            Epi::Gelu => (1, std::ptr::null()),
            Epi::Resid(g) => (2, g),
            Epi::Silu => (3, std::ptr::null()),
            Epi::GeluErf => (4, std::ptr::null()),
        }
    }
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
    s: Option<&'static Silo>,
    g: G,
}

// SAFETY: the GPU handle is a queue; the engine calls through &self from one thread at a time.
unsafe impl Send for K {}
unsafe impl Sync for K {}

impl K {
    pub fn new(gpu: &Gpu) -> Result<K> {
        Ok(K { k: api()?, s: silo()?, g: gpu.raw() })
    }

    pub fn silo(&self) -> Option<&'static Silo> {
        self.s
    }

    /// The silo's product of x [m, k] half and w [n, k] half (+ bias) with the epilogue into out (out_dt): false when
    /// there is no silo or it does not take the epilogue (the caller runs hi3d.h's kernels instead)
    #[allow(clippy::too_many_arguments)]
    pub fn linear_epi(&self, x: P, m: usize, k: usize, w: P, n: usize, bias: *const f32, out: M, out_dt: Dt, epi: Epi) -> Result<bool> {
        let Some(s) = self.s else { return Ok(false) };
        let (e, gate) = epi.code();
        // SAFETY: a plain query.
        if unsafe { (s.linear_supported)(e, d(out_dt)) } == 0 {
            return Ok(false);
        }
        // SAFETY: sizes as named (h3s.h).
        check(unsafe { (s.linear)(self.g, x, m as i64, k as i64, w, n as i64, bias, out, d(out_dt), e, gate) }, "linear (silo)")?;
        Ok(true)
    }

    /// out [m, n] float32 = x [m, k] half . w [n, k] half ^T + bias
    #[allow(clippy::too_many_arguments)]
    pub fn linear_f32(&self, x: P, m: usize, k: usize, w: P, n: usize, bias: *const f32, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.linear_f32)(self.g, x, m as i64, k as i64, w, n as i64, bias, out) }, "linear (float32 out)")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn ln_mod(&self, x: P, x_dt: Dt, m: usize, c: usize, eps: f32, w: *const f32, b: *const f32, scale: *const f32, shift: *const f32, out: M,
                  out_dt: Dt) -> Result<()> {
        if let Some(s) = self.s {
            // SAFETY: a plain query; then sizes as named (h3s.h).
            if unsafe { (s.ln_mod_supported)(c as i64, d(x_dt), d(out_dt)) } != 0 {
                return check(unsafe { (s.ln_mod)(self.g, x, d(x_dt), m as i64, c as i64, eps, w, b, scale, shift, out, d(out_dt)) }, "layer norm (silo)");
            }
        }
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.ln_mod)(self.g, x, d(x_dt), m as i64, c as i64, eps, w, b, scale, shift, out, d(out_dt)) }, "layer norm")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rope_rms(&self, x: M, m: usize, h: usize, dd: usize, stride: usize, cs: *const f32, gamma: *const f32) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.rope_rms)(self.g, x, m as i64, h as i64, dd as i64, stride as i64, cs, gamma) }, "rope + rms norm")
    }
    pub fn gate_add(&self, x: *mut f32, m: usize, c: usize, h: P, h_dt: Dt, gate: *const f32) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.gate_add)(self.g, x, m as i64, c as i64, h, d(h_dt), gate) }, "gated add")
    }
    pub fn add_row(&self, x: *mut f32, m: usize, c: usize, row: *const f32) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.add_row)(self.g, x, m as i64, c as i64, row) }, "row add")
    }
    pub fn convert(&self, src: P, src_dt: Dt, n: usize, dst: M, dst_dt: Dt) -> Result<()> {
        // SAFETY: n values each.
        check(unsafe { (self.k.convert)(self.g, src, d(src_dt), n as i64, dst, d(dst_dt)) }, "convert")
    }
    /// act 0 silu, 1 quick_gelu, 2 tanh
    pub fn act(&self, x: M, dt: Dt, n: usize, act: i32) -> Result<()> {
        // SAFETY: n values.
        check(unsafe { (self.k.act)(self.g, x, d(dt), n as i64, act) }, "activation")
    }
    pub fn add(&self, x: M, x_dt: Dt, y: P, y_dt: Dt, n: usize) -> Result<()> {
        // SAFETY: n values each.
        check(unsafe { (self.k.add)(self.g, x, d(x_dt), y, d(y_dt), n as i64) }, "add")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn put_cols(&self, src: P, src_dt: Dt, n: usize, c: usize, dst: M, dst_dt: Dt, ldd: usize, off: usize) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.put_cols)(self.g, src, d(src_dt), n as i64, c as i64, dst, d(dst_dt), ldd as i64, off as i64) }, "column copy")
    }
    pub fn gather27(&self, feats: P, c: usize, nmap: I, row0: usize, rows: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.gather27)(self.g, feats, c as i64, nmap, row0 as i64, rows as i64, out) }, "sparse gather")
    }
    pub fn gather_rows(&self, src: P, dt: Dt, c: usize, idx: I, n: usize, out: M) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.gather_rows)(self.g, src, d(dt), c as i64, idx, n as i64, out) }, "row gather")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn seg_mean(&self, x: P, x_dt: Dt, c: usize, ptr: I, idx: I, p: usize, out: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.seg_mean)(self.g, x, d(x_dt), c as i64, ptr, idx, p as i64, out) }, "segment mean")
    }
    pub fn gn_sums(&self, x: *const f32, n: usize, c: usize, groups: usize, rows: usize, part: *mut f32) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.gn_sums)(self.g, x, n as i64, c as i64, groups as i64, rows as i64, part) }, "group norm sums")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gn_swish(&self, x: *const f32, n: usize, c: usize, groups: usize, mean: *const f32, inv: *const f32, w: *const f32, b: *const f32, out: M,
                    out_dt: Dt) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.gn_swish)(self.g, x, n as i64, c as i64, groups as i64, mean, inv, w, b, out, d(out_dt)) }, "group norm")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn win_attn(&self, qkv: P, windows: usize, s: usize, h: usize, dd: usize, scale: f32, out: M) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.win_attn)(self.g, qkv, windows as i64, s as i64, h as i64, dd as i64, scale, out) }, "window attention")
    }
    /// Heads of sd features (rows ss apart) into heads of dd (rows ds apart), scaled, zero-padded
    #[allow(clippy::too_many_arguments)]
    pub fn copy_heads(&self, src: P, ss: usize, sd: usize, m: usize, h: usize, dst: M, ds: usize, dd: usize, scale: f32) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.copy_heads)(self.g, src, ss as i64, sd as i64, m as i64, h as i64, dst, ds as i64, dd as i64, scale) }, "head copy")
    }
    pub fn child_nmap(&self, pnmap: I, kids: I, parent: I, sub: I, n: usize, out: *mut i32) -> Result<()> {
        // SAFETY: sizes as named (hi3d.h).
        check(unsafe { (self.k.child_nmap)(self.g, pnmap, kids, parent, sub, n as i64, out) }, "child neighbour maps")
    }
}
