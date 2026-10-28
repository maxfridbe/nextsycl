//! This engine's own kernels (`kernels/model3d/skintokens/skintokens.h`), bound from the 3D kernel library by name, with
//! checked calls on device buffers (the sizes each names are the caller's to hold; skintokens.h says them).
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
    rms: unsafe extern "C" fn(G, F, i64, i64, i64, f32, F, O) -> c_int,
    act: unsafe extern "C" fn(G, O, i64, c_int) -> c_int,
    add: unsafe extern "C" fn(G, O, F, i64) -> c_int,
    swiglu: unsafe extern "C" fn(G, F, i64, i64, O) -> c_int,
    rowdot: unsafe extern "C" fn(G, F, i64, i64, F, f32, c_int, O) -> c_int,
    gather: unsafe extern "C" fn(G, F, i64, I, i64, i64, O) -> c_int,
    qkv_post: unsafe extern "C" fn(G, O, i64, i64, i64, i64, f32, F, F, I, F, I, O, O) -> c_int,
    attention: unsafe extern "C" fn(G, F, i64, i64, F, i64, i64, F, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, c_int, O, i64, O, i64) -> c_int,
    attn_slots: unsafe extern "C" fn(G, F, i64, F, F, I, i64, i64, i64, i64, i64, i64, f32, O) -> c_int,
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
                // SAFETY: the symbol is declared in skintokens.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            ln: sym!("ns_model3d_skintokens_ln"),
            rms: sym!("ns_model3d_skintokens_rms"),
            act: sym!("ns_model3d_skintokens_act"),
            add: sym!("ns_model3d_skintokens_add"),
            swiglu: sym!("ns_model3d_skintokens_swiglu"),
            rowdot: sym!("ns_model3d_skintokens_rowdot"),
            gather: sym!("ns_model3d_skintokens_gather"),
            qkv_post: sym!("ns_model3d_skintokens_qkv_post"),
            attention: sym!("ns_model3d_skintokens_attention"),
            attn_slots: sym!("ns_model3d_skintokens_attn_slots"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// The silo's kernels (kernels/model3d/skintokens/silo/silo.h, dist/silo/libnextsycl-skintokens.so)
#[allow(clippy::type_complexity)]
pub struct Silo {
    pub path: std::path::PathBuf,
    attention_supported: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> c_int,
    attention: unsafe extern "C" fn(G, F, i64, i64, F, i64, i64, F, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, c_int, O, i64) -> c_int,
    slots_supported: unsafe extern "C" fn(i64, i64, i64, i64) -> c_int,
    slots: unsafe extern "C" fn(G, F, i64, F, F, I, i64, i64, i64, i64, i64, i64, f32, O) -> c_int,
    gemv_supported: unsafe extern "C" fn(i64, i64, i64) -> c_int,
    gemv: unsafe extern "C" fn(G, F, *const u16, i64, i64, i64, O) -> c_int,
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
            attention_supported: sym!("ns_model3d_skintokens_silo_attention_supported"),
            attention: sym!("ns_model3d_skintokens_silo_attention"),
            slots_supported: sym!("ns_model3d_skintokens_silo_attn_slots_supported"),
            slots: sym!("ns_model3d_skintokens_silo_attn_slots"),
            gemv_supported: sym!("ns_model3d_skintokens_silo_gemv_supported"),
            gemv: sym!("ns_model3d_skintokens_silo_gemv"),
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
pub const SIGMOID: i32 = 1;

/// The kernels on one GPU
pub struct K {
    k: &'static Api,
    g: G,
    silo: Option<&'static Silo>,
}

// SAFETY: a GPU handle (its queue is thread-safe) and function pointers.
unsafe impl Send for K {}
unsafe impl Sync for K {}

/// A key / value head layout of an attention call: `(buffer, row stride, head stride)`
pub type Strided<'a> = (&'a DevBuf, usize, usize);

impl K {
    pub fn new(gpu: &Gpu) -> Result<K> {
        Ok(K { k: api()?, g: gpu.raw(), silo: silo()? })
    }

    pub fn silo(&self) -> Option<&'static Silo> {
        self.silo
    }

    /// out [m, n] = x [m, k] . W^T for W [n, k] held as bfloat16 (`w`, device memory) when the silo takes the shapes: true if it ran
    pub fn gemv(&self, x: &DevBuf, m: usize, w: &DevBuf, n: usize, k: usize, out: &DevBuf) -> Result<bool> {
        let Some(s) = self.silo.filter(|s| unsafe { (s.gemv_supported)(m as i64, n as i64, k as i64) } == 1) else { return Ok(false) };
        // SAFETY: x holds m rows of k, w n rows of k bfloat16 values, out m n floats.
        check(unsafe { (s.gemv)(self.g, x.fp(), w.ptr() as *const u16, m as i64, n as i64, k as i64, out.fp()) }, "decode product (silo)")?;
        Ok(true)
    }

    /// out [m, c] = layer norm of the rows of x (stride x_row) (* w + b)
    #[allow(clippy::too_many_arguments)]
    pub fn ln(&self, x: &DevBuf, x_row: usize, m: usize, c: usize, eps: f32, w: Option<&DevBuf>, b: Option<&DevBuf>, out: &DevBuf) -> Result<()> {
        // SAFETY: x holds m rows of x_row, out m c; w, b c floats.
        check(unsafe { (self.k.ln)(self.g, x.fp(), x_row as i64, m as i64, c as i64, eps, p(w), p(b), out.fp()) }, "layer norm")
    }

    /// out [m, c] = RMS norm of the rows of x (stride x_row) * w
    #[allow(clippy::too_many_arguments)]
    pub fn rms(&self, x: &DevBuf, x_row: usize, m: usize, c: usize, eps: f32, w: &DevBuf, out: &DevBuf) -> Result<()> {
        // SAFETY: x holds m rows of x_row, out m c; w c floats.
        check(unsafe { (self.k.rms)(self.g, x.fp(), x_row as i64, m as i64, c as i64, eps, w.fp(), out.fp()) }, "RMS norm")
    }

    pub fn act(&self, x: &DevBuf, n: usize, kind: i32) -> Result<()> {
        // SAFETY: x holds n floats.
        check(unsafe { (self.k.act)(self.g, x.fp(), n as i64, kind) }, "activation")
    }

    pub fn add(&self, x: &DevBuf, y: &DevBuf, n: usize) -> Result<()> {
        // SAFETY: x and y hold n floats.
        check(unsafe { (self.k.add)(self.g, x.fp(), y.fp(), n as i64) }, "add")
    }

    /// out [m, i] = silu(gate) * up of the rows gu [m, 2 i]
    pub fn swiglu(&self, gu: &DevBuf, m: usize, i: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: gu holds m 2 i floats, out m i.
        check(unsafe { (self.k.swiglu)(self.g, gu.fp(), m as i64, i as i64, out.fp()) }, "SwiGLU")
    }

    /// out [m] = x [m, c] . w + b (then sigmoid when `act` is SIGMOID)
    #[allow(clippy::too_many_arguments)]
    pub fn rowdot(&self, x: &DevBuf, m: usize, c: usize, w: &DevBuf, b: f32, act: i32, out: &DevBuf) -> Result<()> {
        // SAFETY: x holds m c floats, w c, out m.
        check(unsafe { (self.k.rowdot)(self.g, x.fp(), m as i64, c as i64, w.fp(), b, act, out.fp()) }, "row dot")
    }

    pub fn gather(&self, x: &DevBuf, x_row: usize, idx: &DevBuf, r: usize, c: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: idx holds r int32 rows of x, out r c floats.
        check(unsafe { (self.k.gather)(self.g, x.fp(), x_row as i64, idx.ptr() as I, r as i64, c as i64, out.fp()) }, "gather")
    }

    /// The decoder's attention input (see skintokens.h): the heads of qkv [m, (hq + 2 hkv) d] normed and rotated, the keys and
    /// values written to the pool rows `slot`
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_post(&self, qkv: &DevBuf, m: usize, hq: usize, hkv: usize, d: usize, eps: f32, qw: &DevBuf, kw: &DevBuf, pos: &DevBuf, inv_freq: &DevBuf,
                    slot: &DevBuf, kpool: &DevBuf, vpool: &DevBuf) -> Result<()> {
        // SAFETY: qkv holds m rows of (hq + 2 hkv) d; pos and slot m int32; the pools hold every slot named, rows of hkv d.
        check(unsafe {
            (self.k.qkv_post)(self.g, qkv.fp(), m as i64, hq as i64, hkv as i64, d as i64, eps, qw.fp(), kw.fp(), pos.ptr() as I, inv_freq.fp(),
                              slot.ptr() as I, kpool.fp(), vpool.fp())
        }, "q / k norm and rotary embedding")
    }

    /// Attention (see skintokens.h): `(buffer, row stride, head stride)` for q, k, v; B sequences of Sq queries over Skv keys, h
    /// query heads over hkv key / value heads
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, q: Strided, k: Strided, v: Strided, b: usize, sq: usize, skv: usize, h: usize, hkv: usize, d: usize, dv: usize, scale: f32,
                     causal: bool, scratch: &DevBuf, out: (&DevBuf, usize)) -> Result<()> {
        // SAFETY (both): the strides and counts the caller names stay inside its buffers; the scratch's size is passed.
        if let Some(s) = self.silo.filter(|s| unsafe { (s.attention_supported)(d as i64, dv as i64, h as i64, hkv as i64, skv as i64) } == 1) {
            return check(unsafe {
                (s.attention)(self.g, q.0.fp(), q.1 as i64, q.2 as i64, k.0.fp(), k.1 as i64, k.2 as i64, v.0.fp(), v.1 as i64, v.2 as i64, b as i64,
                              sq as i64, skv as i64, h as i64, hkv as i64, d as i64, dv as i64, scale, causal as c_int, out.0.fp(), out.1 as i64)
            }, "attention (silo)");
        }
        check(unsafe {
            (self.k.attention)(self.g, q.0.fp(), q.1 as i64, q.2 as i64, k.0.fp(), k.1 as i64, k.2 as i64, v.0.fp(), v.1 as i64, v.2 as i64, b as i64,
                               sq as i64, skv as i64, h as i64, hkv as i64, d as i64, dv as i64, scale, causal as c_int, scratch.fp(),
                               scratch.floats() as i64, out.0.fp(), out.1 as i64)
        }, "attention")
    }

    /// One decode step's attention of `b` rows over the slot pool (slots: int32 [b, stride], `len` each)
    #[allow(clippy::too_many_arguments)]
    pub fn attn_slots(&self, q: &DevBuf, q_row: usize, kpool: &DevBuf, vpool: &DevBuf, slots: &DevBuf, stride: usize, len: usize, b: usize, h: usize,
                      hkv: usize, d: usize, scale: f32, out: &DevBuf) -> Result<()> {
        // SAFETY (both): q holds b rows of q_row, out b h d floats; the slots name rows of the pools.
        if let Some(s) = self.silo.filter(|s| unsafe { (s.slots_supported)(d as i64, len as i64, h as i64, hkv as i64) } == 1) {
            return check(unsafe {
                (s.slots)(self.g, q.fp(), q_row as i64, kpool.fp(), vpool.fp(), slots.ptr() as I, stride as i64, len as i64, b as i64, h as i64, hkv as i64,
                          d as i64, scale, out.fp())
            }, "slot attention (silo)");
        }
        check(unsafe {
            (self.k.attn_slots)(self.g, q.fp(), q_row as i64, kpool.fp(), vpool.fp(), slots.ptr() as I, stride as i64, len as i64, b as i64, h as i64,
                                hkv as i64, d as i64, scale, out.fp())
        }, "slot attention")
    }
}
