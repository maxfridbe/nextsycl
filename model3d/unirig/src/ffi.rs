//! This engine's own kernels (`kernels/model3d/unirig/unirig.h`), bound from the 3D kernel library by name, and its
//! silo's (`kernels/model3d/unirig/silo/silo.h`) when built, with checked calls on device buffers (the sizes each names
//! are the caller's to hold; unirig.h says them).
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
    ln: unsafe extern "C" fn(G, F, i64, i64, i64, f32, F, F, O) -> c_int,
    head_norm: unsafe extern "C" fn(G, O, i64, i64, i64, i64, i64, f32) -> c_int,
    act: unsafe extern "C" fn(G, O, i64, c_int) -> c_int,
    round_half: unsafe extern "C" fn(G, O, i64) -> c_int,
    add: unsafe extern "C" fn(G, O, F, i64) -> c_int,
    affine: unsafe extern "C" fn(G, O, i64, i64, F, F) -> c_int,
    attention: unsafe extern "C" fn(G, F, i64, i64, F, i64, i64, F, i64, i64, i64, i64, i64, i64, i64, i64, f32, c_int, O, i64, O, i64) -> c_int,
    attn_slots: unsafe extern "C" fn(G, F, F, F, I, i64, i64, i64, i64, i64, f32, O) -> c_int,
    gather: unsafe extern "C" fn(G, F, i64, I, i64, i64, O) -> c_int,
    scatter: unsafe extern "C" fn(G, F, I, i64, i64, O) -> c_int,
    im2col: unsafe extern "C" fn(G, F, i64, I, i64, i64, i64, O) -> c_int,
    bone_attn: unsafe extern "C" fn(G, F, F, i64, i64, i64, i64, f32, O) -> c_int,
    skin_in: unsafe extern "C" fn(G, F, F, i64, i64, F, F, F, F, F, F, O) -> c_int,
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
                // SAFETY: the symbol is declared in unirig.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            ln: sym!("ns_model3d_unirig_ln"),
            head_norm: sym!("ns_model3d_unirig_head_norm"),
            act: sym!("ns_model3d_unirig_act"),
            round_half: sym!("ns_model3d_unirig_round_half"),
            add: sym!("ns_model3d_unirig_add"),
            affine: sym!("ns_model3d_unirig_affine"),
            attention: sym!("ns_model3d_unirig_attention"),
            attn_slots: sym!("ns_model3d_unirig_attn_slots"),
            gather: sym!("ns_model3d_unirig_gather"),
            scatter: sym!("ns_model3d_unirig_scatter"),
            im2col: sym!("ns_model3d_unirig_im2col"),
            bone_attn: sym!("ns_model3d_unirig_bone_attn"),
            skin_in: sym!("ns_model3d_unirig_skin_in"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// The silo's kernels (kernels/model3d/unirig/silo/silo.h, dist/silo/libnextsycl-unirig.so)
#[allow(clippy::type_complexity)]
pub struct Silo {
    pub path: std::path::PathBuf,
    attention_supported: unsafe extern "C" fn(i64, i64, c_int) -> c_int,
    attention: unsafe extern "C" fn(G, F, i64, i64, F, i64, i64, F, i64, i64, i64, i64, i64, i64, i64, i64, f32, c_int, O, i64, O, i64) -> c_int,
    attn_slots_supported: unsafe extern "C" fn(i64, i64) -> c_int,
    attn_slots: unsafe extern "C" fn(G, F, F, F, I, i64, i64, i64, i64, i64, f32, O) -> c_int,
    bone_attn_supported: unsafe extern "C" fn(i64, i64) -> c_int,
    ln_supported: unsafe extern "C" fn(i64) -> c_int,
    ln: unsafe extern "C" fn(G, F, i64, i64, i64, f32, F, F, c_int, O) -> c_int,
    bone_attn: unsafe extern "C" fn(G, F, F, i64, i64, i64, i64, f32, O) -> c_int,
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
            attention_supported: sym!("ns_model3d_unirig_silo_attention_supported"),
            attention: sym!("ns_model3d_unirig_silo_attention"),
            attn_slots_supported: sym!("ns_model3d_unirig_silo_attn_slots_supported"),
            attn_slots: sym!("ns_model3d_unirig_silo_attn_slots"),
            bone_attn_supported: sym!("ns_model3d_unirig_silo_bone_attn_supported"),
            ln_supported: sym!("ns_model3d_unirig_silo_ln_supported"),
            ln: sym!("ns_model3d_unirig_silo_ln"),
            bone_attn: sym!("ns_model3d_unirig_silo_bone_attn"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| Error(e.clone()))
}

fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("{what}: {why}")))
}

fn p(b: Option<&DevBuf>) -> F {
    b.map_or(std::ptr::null(), |b| b.fp() as F)
}

pub const GELU: i32 = 0;
pub const RELU: i32 = 1;

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

    pub fn silo(&self) -> Option<&'static Silo> {
        self.silo
    }

    /// out [m, c] = layer norm of the rows of x (stride x_row) (* w + b)
    #[allow(clippy::too_many_arguments)]
    pub fn ln(&self, x: &DevBuf, x_row: usize, m: usize, c: usize, eps: f32, w: Option<&DevBuf>, b: Option<&DevBuf>, out: &DevBuf) -> Result<()> {
        // SAFETY (both): x holds m rows of x_row, out m c; w, b c floats.
        if let Some(s) = self.silo.filter(|s| unsafe { (s.ln_supported)(c as i64) } == 1) {
            return check(unsafe { (s.ln)(self.g, x.fp(), x_row as i64, m as i64, c as i64, eps, p(w), p(b), -1, out.fp()) }, "layer norm (silo)");
        }
        check(unsafe { (self.k.ln)(self.g, x.fp(), x_row as i64, m as i64, c as i64, eps, p(w), p(b), out.fp()) }, "layer norm")
    }

    /// GELU(layer norm of x) (rows of `c`, in place when out is x): one pass in the silo, else the norm and the GELU
    #[allow(clippy::too_many_arguments)]
    pub fn ln_gelu(&self, x: &DevBuf, m: usize, c: usize, eps: f32, w: Option<&DevBuf>, b: Option<&DevBuf>, out: &DevBuf) -> Result<()> {
        // SAFETY: x and out hold m c floats; w, b c floats.
        if let Some(s) = self.silo.filter(|s| unsafe { (s.ln_supported)(c as i64) } == 1) {
            return check(unsafe { (s.ln)(self.g, x.fp(), c as i64, m as i64, c as i64, eps, p(w), p(b), GELU, out.fp()) }, "layer norm, GELU (silo)");
        }
        self.ln(x, c, m, c, eps, w, b, out)?;
        self.act(out, m * c, GELU)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn head_norm(&self, x: &DevBuf, m: usize, row: usize, col: usize, heads: usize, d: usize, eps: f32) -> Result<()> {
        // SAFETY: x holds m rows of `row`, the heads inside them.
        check(unsafe { (self.k.head_norm)(self.g, x.fp(), m as i64, row as i64, col as i64, heads as i64, d as i64, eps) }, "head norm")
    }

    pub fn act(&self, x: &DevBuf, n: usize, kind: i32) -> Result<()> {
        // SAFETY: x holds n floats.
        check(unsafe { (self.k.act)(self.g, x.fp(), n as i64, kind) }, "activation")
    }

    /// x rounded to half and back, in place (n values)
    pub fn round_half(&self, x: &DevBuf, n: usize) -> Result<()> {
        // SAFETY: x holds n floats.
        check(unsafe { (self.k.round_half)(self.g, x.fp(), n as i64) }, "rounding to half")
    }

    pub fn add(&self, x: &DevBuf, y: &DevBuf, n: usize) -> Result<()> {
        // SAFETY: x and y hold n floats.
        check(unsafe { (self.k.add)(self.g, x.fp(), y.fp(), n as i64) }, "add")
    }

    pub fn affine(&self, x: &DevBuf, m: usize, c: usize, scale: &DevBuf, shift: &DevBuf) -> Result<()> {
        // SAFETY: x holds m c floats, scale and shift c.
        check(unsafe { (self.k.affine)(self.g, x.fp(), m as i64, c as i64, scale.fp(), shift.fp()) }, "affine")
    }

    /// Attention (see unirig.h): `(buffer, row stride, head stride)` for q, k, v; B sequences of Sq queries over Skv keys
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

    /// One decode step's attention of `b` rows over the slot pool (slots: int32 [b, stride], `len` each)
    #[allow(clippy::too_many_arguments)]
    pub fn attn_slots(&self, q: &DevBuf, kpool: &DevBuf, vpool: &DevBuf, slots: &DevBuf, stride: usize, len: usize, b: usize, h: usize, d: usize,
                      scale: f32, out: &DevBuf) -> Result<()> {
        // SAFETY (both): q and out hold b h d floats; the slots name rows of the pools.
        if let Some(s) = self.silo.filter(|s| unsafe { (s.attn_slots_supported)(d as i64, len as i64) } == 1) {
            return check(unsafe {
                (s.attn_slots)(self.g, q.fp(), kpool.fp(), vpool.fp(), slots.ptr() as I, stride as i64, len as i64, b as i64, h as i64, d as i64, scale,
                               out.fp())
            }, "slot attention (silo)");
        }
        check(unsafe {
            (self.k.attn_slots)(self.g, q.fp(), kpool.fp(), vpool.fp(), slots.ptr() as I, stride as i64, len as i64, b as i64, h as i64, d as i64, scale,
                                out.fp())
        }, "slot attention")
    }

    pub fn gather(&self, x: &DevBuf, x_row: usize, idx: &DevBuf, r: usize, c: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: idx holds r int32 rows of x, out r c floats.
        check(unsafe { (self.k.gather)(self.g, x.fp(), x_row as i64, idx.ptr() as I, r as i64, c as i64, out.fp()) }, "gather")
    }

    pub fn scatter(&self, x: &DevBuf, idx: &DevBuf, r: usize, c: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: idx holds r int32 rows of out, x r c floats.
        check(unsafe { (self.k.scatter)(self.g, x.fp(), idx.ptr() as I, r as i64, c as i64, out.fp()) }, "scatter")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn im2col(&self, x: &DevBuf, c: usize, nbr: &DevBuf, k: usize, r0: usize, r: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: nbr holds int32 [N, k] rows of x [N, c] (or -1); out r k c floats.
        check(unsafe { (self.k.im2col)(self.g, x.fp(), c as i64, nbr.ptr() as I, k as i64, r0 as i64, r as i64, out.fp()) }, "im2col")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bone_attn(&self, q: &DevBuf, k: &DevBuf, r: usize, j: usize, h: usize, d: usize, scale: f32, out: &DevBuf) -> Result<()> {
        // SAFETY (both): q holds r h d floats, k j h d, out r j h.
        if let Some(s) = self.silo.filter(|s| unsafe { (s.bone_attn_supported)(j as i64, d as i64) } == 1) {
            return check(unsafe { (s.bone_attn)(self.g, q.fp(), k.fp(), r as i64, j as i64, h as i64, d as i64, scale, out.fp()) }, "bone attention (silo)");
        }
        check(unsafe { (self.k.bone_attn)(self.g, q.fp(), k.fp(), r as i64, j as i64, h as i64, d as i64, scale, out.fp()) }, "bone attention")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn skin_in(&self, a: &DevBuf, vox: &DevBuf, rj: usize, h: usize, aw: &DevBuf, ab: &DevBuf, ew: &DevBuf, eb: &DevBuf, vw: &DevBuf, vb: &DevBuf,
                   out: &DevBuf) -> Result<()> {
        // SAFETY: a holds rj h floats, vox rj, the vectors h, out rj 2 h.
        check(unsafe {
            (self.k.skin_in)(self.g, a.fp(), vox.fp(), rj as i64, h as i64, aw.fp(), ab.fp(), ew.fp(), eb.fp(), vw.fp(), vb.fp(), out.fp())
        }, "skin input")
    }
}
