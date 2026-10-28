//! This engine's own kernels (`kernels/model3d/unimate/unimate.h`), bound from the 3D kernel library by name, with
//! checked calls on device buffers (the sizes each names are the caller's to hold; the doc of each says them).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{DevBuf, Error, Gpu, Result};

type G = *mut c_void;
type F = *const f32;
type O = *mut f32;

#[allow(clippy::type_complexity)]
struct Api {
    rms: unsafe extern "C" fn(G, F, i64, i64, i64, i64, f32, F, F, F, O, i64) -> c_int,
    gated_add: unsafe extern "C" fn(G, O, F, i64, i64, F) -> c_int,
    act: unsafe extern "C" fn(G, O, i64, c_int) -> c_int,
    glu: unsafe extern "C" fn(G, F, i64, i64, c_int, O) -> c_int,
    rope: unsafe extern "C" fn(G, O, i64, i64, i64, i64, i64, i64, i64, i64, F, F) -> c_int,
    attention: unsafe extern "C" fn(G, F, i64, F, i64, F, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, F, O, i64) -> c_int,
    copy_rows: unsafe extern "C" fn(G, F, i64, i64, O, i64, i64, i64, i64) -> c_int,
    cat2: unsafe extern "C" fn(G, F, i64, F, i64, i64, O) -> c_int,
    tokens: unsafe extern "C" fn(G, F, F, F, F, F, i64, i64, i64, O) -> c_int,
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
                // SAFETY: the symbol is declared in unimate.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            rms: sym!("ns_model3d_unimate_rms"),
            gated_add: sym!("ns_model3d_unimate_gated_add"),
            act: sym!("ns_model3d_unimate_act"),
            glu: sym!("ns_model3d_unimate_glu"),
            rope: sym!("ns_model3d_unimate_rope"),
            attention: sym!("ns_model3d_unimate_attention"),
            copy_rows: sym!("ns_model3d_unimate_copy_rows"),
            cat2: sym!("ns_model3d_unimate_cat2"),
            tokens: sym!("ns_model3d_unimate_tokens"),
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
fn p(b: Option<&DevBuf>) -> F {
    b.map_or(std::ptr::null(), |b| b.fp() as F)
}

/// Activations (`act`)
pub const SILU: i32 = 0;
pub const GELU: i32 = 1;

/// The gated units (`glu`): SwiGLU, and T5's gated GELU (the tanh form)
pub const SWIGLU: i32 = 0;
pub const GATED_GELU: i32 = 1;

/// A buffer and the floats its rows are apart: a view of rows
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub buf: &'a DevBuf,
    /// the first float
    pub at: usize,
    /// floats a row
    pub row: usize,
}

impl<'a> Rows<'a> {
    pub fn new(buf: &'a DevBuf, row: usize) -> Rows<'a> {
        Rows { buf, at: 0, row }
    }
    fn ptr(&self) -> F {
        // SAFETY: an offset inside the buffer (the caller's sizes); a pointer is only formed here, read by the kernel.
        unsafe { self.buf.fp().add(self.at) as F }
    }
}

/// The attention's geometry (unimate.h)
#[derive(Clone, Copy)]
pub struct Attn {
    pub groups: usize,
    pub sq: usize,
    pub skv: usize,
    pub heads: usize,
    pub d: usize,
    pub q_gs: usize,
    pub q_ss: usize,
    pub kv_gs: usize,
    pub kv_ss: usize,
    pub nvalid: usize,
    pub scale: f32,
}

/// The kernels on one GPU
pub struct K {
    k: &'static Api,
    g: G,
}

// SAFETY: a GPU handle (its queue is thread-safe) and function pointers.
unsafe impl Send for K {}
unsafe impl Sync for K {}

impl K {
    pub fn new(gpu: &Gpu) -> Result<K> {
        Ok(K { k: api()?, g: gpu.raw() })
    }

    /// RMS norm of the rows (m, h) of `x` (C values each) times `w` [C], modulated when `scale` / `shift` are given
    #[allow(clippy::too_many_arguments)]
    pub fn rms(&self, x: Rows, m: usize, h: usize, c: usize, eps: f32, w: &DevBuf, modulation: Option<(Rows, Rows)>, out: Rows) -> Result<()> {
        let (sc, sh) = modulation.map_or((std::ptr::null(), std::ptr::null()), |(a, b)| (a.ptr(), b.ptr()));
        // SAFETY: x and out hold m rows of h c floats; w c floats; the modulation vectors c.
        check(unsafe { (self.k.rms)(self.g, x.ptr(), x.row as i64, m as i64, h as i64, c as i64, eps, w.fp(), sc, sh, out.ptr() as O, out.row as i64) }, "rms norm")
    }

    /// x [m, c] += gate * y (gate: c floats, or None)
    pub fn gated_add(&self, x: &DevBuf, y: &DevBuf, m: usize, c: usize, gate: Option<Rows>) -> Result<()> {
        // SAFETY: x and y hold m c floats, the gate c.
        check(unsafe { (self.k.gated_add)(self.g, x.fp(), y.fp(), m as i64, c as i64, gate.map_or(std::ptr::null(), |r| r.ptr())) }, "gated add")
    }

    pub fn act(&self, x: &DevBuf, n: usize, kind: i32) -> Result<()> {
        // SAFETY: x holds n floats.
        check(unsafe { (self.k.act)(self.g, x.fp(), n as i64, kind) }, "activation")
    }

    pub fn glu(&self, x: &DevBuf, m: usize, c: usize, kind: i32, out: &DevBuf) -> Result<()> {
        // SAFETY: x holds m 2c floats, out m c.
        check(unsafe { (self.k.glu)(self.g, x.fp(), m as i64, c as i64, kind, out.fp()) }, "gated unit")
    }

    /// Rotary embedding in place on `nseg` segments `seg` floats apart of m rows of h heads of d
    #[allow(clippy::too_many_arguments)]
    pub fn rope(&self, x: &DevBuf, row: usize, seg: usize, nseg: usize, m: usize, h: usize, d: usize, div: usize, modulus: usize, cos: &DevBuf, sin: &DevBuf) -> Result<()> {
        // SAFETY: x holds the segments' rows; the tables modulus d floats each.
        check(unsafe { (self.k.rope)(self.g, x.fp(), row as i64, seg as i64, nseg as i64, m as i64, h as i64, d as i64, div as i64, modulus as i64, cos.fp(), sin.fp()) }, "rotary embedding")
    }

    /// Attention (unimate.h); `bias` [H, Sq, Skv] is optional
    pub fn attention(&self, q: Rows, k: Rows, v: Rows, a: Attn, bias: Option<&DevBuf>, out: Rows) -> Result<()> {
        // SAFETY: the strides and counts the caller names stay inside its buffers.
        check(unsafe {
            (self.k.attention)(self.g, q.ptr(), q.row as i64, k.ptr(), k.row as i64, v.ptr(), v.row as i64, a.groups as i64, a.sq as i64, a.skv as i64, a.heads as i64,
                               a.d as i64, a.q_gs as i64, a.q_ss as i64, a.kv_gs as i64, a.kv_ss as i64, a.nvalid as i64, a.scale, p(bias), out.ptr() as O, out.row as i64)
        }, "attention")
    }

    /// n rows of c floats: src rows `src_off + r src_stride` to dst rows `dst_off + r dst_stride`
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows(&self, src: &DevBuf, src_off: usize, src_stride: usize, dst: &DevBuf, dst_off: usize, dst_stride: usize, n: usize, c: usize) -> Result<()> {
        // SAFETY: the rows named are inside the buffers.
        check(unsafe { (self.k.copy_rows)(self.g, src.fp(), src_off as i64, src_stride as i64, dst.fp(), dst_off as i64, dst_stride as i64, n as i64, c as i64) }, "copy rows")
    }

    pub fn cat2(&self, a: &DevBuf, c1: usize, b: &DevBuf, c2: usize, m: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: a holds m c1 floats, b m c2, out m (c1 + c2).
        check(unsafe { (self.k.cat2)(self.g, a.fp(), c1 as i64, b.fp(), c2 as i64, m as i64, out.fp()) }, "cat")
    }

    /// The token grid of the denoiser (unimate.h)
    #[allow(clippy::too_many_arguments)]
    pub fn tokens(&self, tpos: &DevBuf, joints: &DevBuf, root: &DevBuf, depth: &DevBuf, name: &DevBuf, f: usize, j: usize, d: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: tpos, depth, name hold j d floats, joints f j d, root f d, out (f + 1) j d.
        check(unsafe { (self.k.tokens)(self.g, tpos.fp(), joints.fp(), root.fp(), depth.fp(), name.fp(), f as i64, j as i64, d as i64, out.fp()) }, "token grid")
    }
}
