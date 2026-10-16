//! This engine's own kernels (`kernels/model3d/animateanymesh/aam.h`), bound from the 3D kernel library by name, with
//! checked calls on device buffers (the sizes each names are the caller's to hold; the doc of each says them).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{DevBuf, Error, Gpu, Result};

type G = *mut c_void;
type F = *const f32;
type O = *mut f32;
type I = *const i32;

#[allow(clippy::type_complexity)]
struct Api {
    ln: unsafe extern "C" fn(G, F, i64, i64, f32, i64, i64, F, F, F, F, F, F, F, F, O) -> c_int,
    gated_add: unsafe extern "C" fn(G, O, F, i64, i64, i64, i64, F, F) -> c_int,
    act: unsafe extern "C" fn(G, O, i64, c_int) -> c_int,
    geglu: unsafe extern "C" fn(G, F, i64, i64, O) -> c_int,
    attention: unsafe extern "C" fn(G, F, i64, i64, F, i64, i64, F, i64, i64, i64, i64, i64, i64, i64, i64, f32, c_int, O, i64, O, i64) -> c_int,
    csr_attention: unsafe extern "C" fn(G, F, F, F, i64, i64, I, I, f32, O) -> c_int,
    fps: unsafe extern "C" fn(G, F, i64, i64, i64, O, *mut i32) -> c_int,
    gather: unsafe extern "C" fn(G, F, I, i64, i64, O) -> c_int,
    flow_in: unsafe extern "C" fn(G, F, F, i64, i64, i64, O) -> c_int,
    cfg: unsafe extern "C" fn(G, F, i64, f32, O) -> c_int,
    axpy: unsafe extern "C" fn(G, F, F, f32, i64, O) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (the 3D library, rebuilt: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in aam.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            ln: sym!("ns_model3d_aam_ln"),
            gated_add: sym!("ns_model3d_aam_gated_add"),
            act: sym!("ns_model3d_aam_act"),
            geglu: sym!("ns_model3d_aam_geglu"),
            attention: sym!("ns_model3d_aam_attention"),
            csr_attention: sym!("ns_model3d_aam_csr_attention"),
            fps: sym!("ns_model3d_aam_fps"),
            gather: sym!("ns_model3d_aam_gather"),
            flow_in: sym!("ns_model3d_aam_flow_in"),
            cfg: sym!("ns_model3d_aam_cfg"),
            axpy: sym!("ns_model3d_aam_axpy"),
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

/// A float buffer's device pointer, or null
pub fn p(b: Option<&DevBuf>) -> F {
    b.map_or(std::ptr::null(), |b| b.fp() as F)
}

/// Activations (`act`)
pub const GELU: i32 = 0;
pub const SILU: i32 = 1;
pub const QUICK_GELU: i32 = 2;

/// Where a layer norm's affine and modulation come from, per stream (each float32 [C] or None)
#[derive(Clone, Copy, Default)]
pub struct Norm<'a> {
    pub w: Option<&'a DevBuf>,
    pub b: Option<&'a DevBuf>,
    pub scale: Option<&'a DevBuf>,
    pub shift: Option<&'a DevBuf>,
}

/// The silo's kernels (kernels/model3d/animateanymesh/silo/silo.h, dist/silo/libnextsycl-animateanymesh.so)
#[allow(clippy::type_complexity)]
pub struct Silo {
    pub path: std::path::PathBuf,
    attention_supported: unsafe extern "C" fn(i64, i64, c_int) -> c_int,
    attention: unsafe extern "C" fn(G, F, i64, i64, F, i64, i64, F, i64, i64, i64, i64, i64, i64, i64, i64, f32, c_int, O, i64, O, i64) -> c_int,
    fps_supported: unsafe extern "C" fn(i64, i64, i64) -> c_int,
    fps: unsafe extern "C" fn(G, F, i64, i64, i64, O, *mut i32) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo (None: the engine's own kernels throughout)
pub fn silo() -> Result<Option<&'static Silo>> {
    static S: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(crate::ARCH).map_err(|e| e.0)? else { return Ok(None) };
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
        Ok(Some(Silo {
            path: lib.path.clone(),
            attention_supported: sym!("ns_model3d_aam_silo_attention_supported"),
            attention: sym!("ns_model3d_aam_silo_attention"),
            fps_supported: sym!("ns_model3d_aam_silo_fps_supported"),
            fps: sym!("ns_model3d_aam_silo_fps"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| Error(e.clone()))
}

/// The kernels on one GPU
pub struct K {
    k: &'static Api,
    g: G,
    silo: Option<&'static Silo>,
}

// SAFETY: a GPU handle (its queue is thread-safe) and function pointers.
unsafe impl Send for K {}
unsafe impl Sync for K {}

impl K {
    pub fn new(gpu: &Gpu) -> Result<K> {
        Ok(K { k: api()?, g: gpu.raw(), silo: silo()? })
    }

    /// The silo in use, if any
    pub fn silo(&self) -> Option<&'static Silo> {
        self.silo
    }

    /// out [M, C] = layer norm of x [M, C], rows (r % s) < t taking `n0`, the others `n1` (s = t: one stream)
    #[allow(clippy::too_many_arguments)]
    pub fn ln(&self, x: &DevBuf, m: usize, c: usize, eps: f32, s: usize, t: usize, n0: Norm, n1: Norm, out: &DevBuf) -> Result<()> {
        // SAFETY: x and out hold m c floats; each norm vector c floats (the caller's).
        check(unsafe {
            (self.k.ln)(self.g, x.fp(), m as i64, c as i64, eps, s as i64, t as i64, p(n0.w), p(n0.b), p(n0.scale), p(n0.shift), p(n1.w), p(n1.b),
                        p(n1.scale), p(n1.shift), out.fp())
        }, "layer norm")
    }

    /// x [M, C] += gate (per stream, as `ln`; None: 1) * y
    #[allow(clippy::too_many_arguments)]
    pub fn gated_add(&self, x: &DevBuf, y: &DevBuf, m: usize, c: usize, s: usize, t: usize, g0: Option<&DevBuf>, g1: Option<&DevBuf>) -> Result<()> {
        // SAFETY: x and y hold m c floats, the gates c.
        check(unsafe { (self.k.gated_add)(self.g, x.fp(), y.fp(), m as i64, c as i64, s as i64, t as i64, p(g0), p(g1)) }, "gated add")
    }

    pub fn act(&self, x: &DevBuf, n: usize, kind: i32) -> Result<()> {
        // SAFETY: x holds n floats.
        check(unsafe { (self.k.act)(self.g, x.fp(), n as i64, kind) }, "activation")
    }

    pub fn geglu(&self, x: &DevBuf, m: usize, c: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: x holds m 2c floats, out m c.
        check(unsafe { (self.k.geglu)(self.g, x.fp(), m as i64, c as i64, out.fp()) }, "geglu")
    }

    /// Attention (see aam.h): `(buffer, row stride, head stride)` for q, k, v; B sequences of Sq queries over Skv keys
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, q: (&DevBuf, usize, usize), k: (&DevBuf, usize, usize), v: (&DevBuf, usize, usize), b: usize, sq: usize, skv: usize,
                     h: usize, d: usize, dv: usize, scale: f32, causal: bool, scratch: &DevBuf, out: (&DevBuf, usize)) -> Result<()> {
        // SAFETY (both): the strides and counts the caller names stay inside its buffers; the scratch's size is passed.
        if let Some(s) = self.silo.filter(|s| unsafe { (s.attention_supported)(d as i64, dv as i64, causal as c_int) } == 1) {
            return check(unsafe {
                (s.attention)(self.g, q.0.fp(), q.1 as i64, q.2 as i64, k.0.fp(), k.1 as i64, k.2 as i64, v.0.fp(), v.1 as i64, v.2 as i64, b as i64,
                              sq as i64, skv as i64, h as i64, d as i64, dv as i64, scale, causal as c_int, scratch.fp(), scratch.floats() as i64,
                              out.0.fp(), out.1 as i64)
            }, "attention (silo)");
        }
        check(unsafe {
            (self.k.attention)(self.g, q.0.fp(), q.1 as i64, q.2 as i64, k.0.fp(), k.1 as i64, k.2 as i64, v.0.fp(), v.1 as i64, v.2 as i64, b as i64,
                               sq as i64, skv as i64, h as i64, d as i64, dv as i64, scale, causal as c_int, scratch.fp(), scratch.floats() as i64,
                               out.0.fp(), out.1 as i64)
        }, "attention")
    }

    /// Attention of each row over its neighbours (`rowptr` [N + 1], `cols` int32 device buffers)
    #[allow(clippy::too_many_arguments)]
    pub fn csr_attention(&self, q: &DevBuf, k: &DevBuf, v: &DevBuf, n: usize, d: usize, rowptr: &DevBuf, cols: &DevBuf, scale: f32, out: &DevBuf) -> Result<()> {
        // SAFETY: q, k, v, out hold n d floats; rowptr n + 1 offsets into cols, whose entries are rows < n.
        check(unsafe {
            (self.k.csr_attention)(self.g, q.fp(), k.fp(), v.fp(), n as i64, d as i64, rowptr.ptr() as I, cols.ptr() as I, scale, out.fp())
        }, "neighbour attention")
    }

    /// Farthest points of x [N, D]: `k` int32 indices into `idx` (dist: N floats of scratch)
    pub fn fps(&self, x: &DevBuf, n: usize, d: usize, k: usize, dist: &DevBuf, idx: &DevBuf) -> Result<()> {
        // SAFETY (both): x holds n d floats, dist n, idx k int32.
        if let Some(s) = self.silo.filter(|s| unsafe { (s.fps_supported)(n as i64, d as i64, k as i64) } == 1) {
            return check(unsafe { (s.fps)(self.g, x.fp(), n as i64, d as i64, k as i64, dist.fp(), idx.ptr().cast()) }, "farthest point sampling (silo)");
        }
        check(unsafe { (self.k.fps)(self.g, x.fp(), n as i64, d as i64, k as i64, dist.fp(), idx.ptr().cast()) }, "farthest point sampling")
    }

    pub fn gather(&self, x: &DevBuf, idx: &DevBuf, r: usize, c: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: idx holds r int32 rows of x (each < its rows), out r c floats.
        check(unsafe { (self.k.gather)(self.g, x.fp(), idx.ptr() as I, r as i64, c as i64, out.fp()) }, "gather")
    }

    pub fn flow_in(&self, f0: &DevBuf, y: &DevBuf, k: usize, c: usize, f: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: f0 holds k f floats, y k c, out 2 k c.
        check(unsafe { (self.k.flow_in)(self.g, f0.fp(), y.fp(), k as i64, c as i64, f as i64, out.fp()) }, "flow input")
    }

    pub fn cfg(&self, flow: &DevBuf, n: usize, scale: f32, out: &DevBuf) -> Result<()> {
        // SAFETY: flow holds 2 n floats, out n.
        check(unsafe { (self.k.cfg)(self.g, flow.fp(), n as i64, scale, out.fp()) }, "guidance")
    }

    pub fn axpy(&self, y: &DevBuf, f: &DevBuf, a: f32, n: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: y, f, out hold n floats.
        check(unsafe { (self.k.axpy)(self.g, y.fp(), f.fp(), a, n as i64, out.fp()) }, "axpy")
    }
}
