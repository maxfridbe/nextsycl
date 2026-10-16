//! This engine's own kernels (`kernels/image/hunyuanimage21/hi21.h`), bound from the image kernel library by name, with
//! checked wrappers on device pointers (the sizes each names are the caller's to hold).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{Error, Gpu, Result};

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

#[allow(clippy::type_complexity)]
struct Api {
    q4_0_half: unsafe extern "C" fn(G, P, i64, M) -> c_int,
    ln_mod: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const f32, *const f32, *const f32, *const f32, M, c_int) -> c_int,
    act: unsafe extern "C" fn(G, M, c_int, i64, c_int) -> c_int,
    geglu: unsafe extern "C" fn(G, P, c_int, i64, i64, M) -> c_int,
    cat_gelu: unsafe extern "C" fn(G, P, i64, i64, P, i64, i64, M) -> c_int,
    attn_small: unsafe extern "C" fn(G, *const f32, i64, *const f32, *const f32, i64, i64, i64, i64, i64, f32, *const f32, c_int, *mut f32) -> c_int,
    linear_f32: unsafe extern "C" fn(G, P, i64, i64, P, i64, *const f32, *mut f32) -> c_int,
    axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    to_half: unsafe extern "C" fn(G, *const f32, M, i64, f32) -> c_int,
    to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    rep_add: unsafe extern "C" fn(G, M, P, i64, i64, i64, i64) -> c_int,
    up_shuffle: unsafe extern "C" fn(G, P, P, i64, i64, i64, i64, i64, M) -> c_int,
    gn_scratch: unsafe extern "C" fn(i64) -> i64,
    group_norm: unsafe extern "C" fn(G, P, i64, i64, i64, *const f32, *const f32, f32, c_int, *mut f32, M) -> c_int,
    transpose: unsafe extern "C" fn(G, P, i64, i64, M) -> c_int,
    softmax_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M) -> c_int,
    to_rgb8: unsafe extern "C" fn(G, P, i64, *mut u8) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the image library, rebuilt: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in hi21.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            q4_0_half: sym!("ns_image_hi21_q4_0_half"),
            ln_mod: sym!("ns_image_hi21_ln_mod"),
            act: sym!("ns_image_hi21_act"),
            geglu: sym!("ns_image_hi21_geglu"),
            cat_gelu: sym!("ns_image_hi21_cat_gelu"),
            attn_small: sym!("ns_image_hi21_attn_small"),
            linear_f32: sym!("ns_image_hi21_linear_f32"),
            axpy: sym!("ns_image_hi21_axpy"),
            to_half: sym!("ns_image_hi21_to_half"),
            to_float: sym!("ns_image_hi21_to_float"),
            rep_add: sym!("ns_image_hi21_rep_add"),
            up_shuffle: sym!("ns_image_hi21_up_shuffle"),
            gn_scratch: sym!("ns_image_hi21_gn_scratch"),
            group_norm: sym!("ns_image_hi21_group_norm"),
            transpose: sym!("ns_image_hi21_transpose"),
            softmax_rows: sym!("ns_image_hi21_softmax_rows"),
            to_rgb8: sym!("ns_image_hi21_to_rgb8"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// The silo's name (`kernels/image/hunyuanimage21/silo` -> `dist/silo/libnextsycl-hunyuanimage21.so`)
pub const SILO: &str = "hunyuanimage21";

/// One product's arguments (silo.h's `ns_hi21_mm`)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mm {
    pub x: P,
    pub lda: i64,
    pub x_bs: i64,
    pub b: i64,
    pub m: i64,
    pub k: i64,
    pub k0: i64,
    pub kn: i64,
    pub n: i64,
    pub w: *const i8,
    pub s: P,
    pub bias: *const f32,
    pub out: M,
    pub ldo: i64,
    pub o_bs: i64,
    pub out_dt: i32,
    pub post: i32,
    pub gate: *const f32,
    pub wscale: *const f32,
    pub xq: *mut i8,
    pub xs: *mut f32,
}

/// The silo's products (silo.h): the DiT's Q8_0 matrices read as int8 in the product, with its fusions
struct Silo {
    path: std::path::PathBuf,
    q8_split: unsafe extern "C" fn(G, P, i64, i64, *mut i8, M) -> c_int,
    q8_rows: unsafe extern "C" fn(G, P, i64, i64, *mut i8, *mut f32) -> c_int,
    linear_supported: unsafe extern "C" fn(G, c_int) -> c_int,
    linear: unsafe extern "C" fn(G, *const Mm) -> c_int,
    gn_supported: unsafe extern "C" fn(i64, i64) -> c_int,
    gn_scratch: unsafe extern "C" fn(i64) -> i64,
    group_norm: unsafe extern "C" fn(G, P, i64, i64, i64, *const f32, *const f32, f32, c_int, *mut f32, M) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo, when built (None: NS_SILO=0, or no file - the engine's own kernels throughout)
fn silo() -> Result<Option<&'static Silo>> {
    static S: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
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
            q8_split: sym!("ns_image_hi21_silo_q8_split"),
            q8_rows: sym!("ns_image_hi21_silo_q8_rows"),
            linear_supported: sym!("ns_image_hi21_silo_linear_supported"),
            linear: sym!("ns_image_hi21_silo_linear"),
            gn_supported: sym!("ns_image_hi21_silo_group_norm_supported"),
            gn_scratch: sym!("ns_image_hi21_silo_gn_scratch"),
            group_norm: sym!("ns_image_hi21_silo_group_norm"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("{what}: {why}")))
}

/// A buffer's type as the kernels name it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum T {
    F32 = 0,
    F16 = 1,
}

/// The activations `act` applies
#[derive(Clone, Copy, Debug)]
pub enum Act {
    Silu = 0,
    GeluTanh = 1,
    GeluErf = 2,
}

/// The kernels on one GPU
#[derive(Clone)]
pub struct K {
    pub gpu: std::sync::Arc<Gpu>,
    k: &'static Api,
    /// the silo's products, when built and supported on this GPU
    silo: Option<&'static Silo>,
    /// whether the silo runs its int8 x int8 products here
    int8: bool,
}

impl K {
    pub fn new(gpu: &std::sync::Arc<Gpu>) -> Result<K> {
        let k = api()?;
        // SAFETY: a capability check on the GPU's handle.
        let silo = silo()?.filter(|s| unsafe { (s.linear_supported)(gpu.raw(), 0) } == 1);
        // SAFETY: as above.
        let int8 = silo.is_some_and(|s| unsafe { (s.linear_supported)(gpu.raw(), 1) } == 1);
        Ok(K { gpu: gpu.clone(), k, silo, int8 })
    }

    /// The silo's library, when its products are used here
    pub fn silo(&self) -> Option<&'static std::path::Path> {
        self.silo.map(|s| s.path.as_path())
    }

    /// A Q8_0 matrix [n, k] into the silo's int8 values [n, k] and half scales [k / 32, n]
    pub fn q8_split(&self, blocks: P, n: usize, k: usize, w: M, s: M) -> Result<()> {
        let silo = self.silo.ok_or_else(|| Error("the silo is not loaded".into()))?;
        // SAFETY: blocks holds n x k / 32 Q8_0 blocks, w n x k bytes, s n x k / 32 halves.
        check(unsafe { (silo.q8_split)(self.g(), blocks, n as i64, k as i64, w.cast(), s) }, "q8_0 split")
    }

    /// Whether the silo's int8 x int8 products run here
    pub fn int8(&self) -> bool {
        self.int8
    }

    /// A Q8_0 matrix [n, k] requantized for the silo's int8 products: int8 [n, k] over a float32 scale a row
    pub fn q8_rows(&self, blocks: P, n: usize, k: usize, w: M, scale: *mut f32) -> Result<()> {
        let silo = self.silo.ok_or_else(|| Error("the silo is not loaded".into()))?;
        // SAFETY: blocks holds n x k / 32 Q8_0 blocks, w n x k bytes, scale n floats.
        check(unsafe { (silo.q8_rows)(self.g(), blocks, n as i64, k as i64, w.cast(), scale) }, "q8_0 to int8 rows")
    }

    /// One of the silo's products (silo.h)
    pub fn mm(&self, a: &Mm) -> Result<()> {
        let silo = self.silo.ok_or_else(|| Error("the silo is not loaded".into()))?;
        // SAFETY: the buffers `a` names hold the rows, columns and strides it gives (the caller's to hold).
        check(unsafe { (silo.linear)(self.g(), a) }, "linear (silo)")
    }

    fn g(&self) -> G {
        self.gpu.raw()
    }

    /// n values of Q4_0 blocks -> half
    pub fn q4_0_half(&self, src: P, n: usize, out: M) -> Result<()> {
        // SAFETY: src holds n / 32 blocks, out n halves.
        check(unsafe { (self.k.q4_0_half)(self.g(), src, n as i64, out) }, "q4_0 to half")
    }

    /// out [m, c] = layernorm(x) (* weight + bias) * (1 + scale) + shift
    #[allow(clippy::too_many_arguments)]
    pub fn ln_mod(&self, x: *const f32, m: usize, c: usize, eps: f32, weight: *const f32, bias: *const f32, scale: *const f32, shift: *const f32,
                  out: M, out_dt: T) -> Result<()> {
        // SAFETY: x [m, c], out [m, c], the vectors [c] or null.
        check(unsafe { (self.k.ln_mod)(self.g(), x, m as i64, c as i64, eps, weight, bias, scale, shift, out, out_dt as c_int) }, "layer norm")
    }

    /// x (dt) in place, n values
    pub fn act(&self, x: M, dt: T, n: usize, a: Act) -> Result<()> {
        // SAFETY: x holds n values.
        check(unsafe { (self.k.act)(self.g(), x, dt as c_int, n as i64, a as c_int) }, "activation")
    }

    /// out [m, c] = gelu_tanh(x[:, :c]) * x[:, c:]
    pub fn geglu(&self, x: P, dt: T, m: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: x [m, 2c], out [m, c].
        check(unsafe { (self.k.geglu)(self.g(), x, dt as c_int, m as i64, c as i64, out) }, "gated gelu")
    }

    /// out [m, a + f] half = [att | gelu_tanh(mlp rows, stride mlp_stride)]
    #[allow(clippy::too_many_arguments)]
    pub fn cat_gelu(&self, att: P, m: usize, a: usize, mlp: P, mlp_stride: usize, f: usize, out: M) -> Result<()> {
        // SAFETY: att [m, a], mlp rows of f values mlp_stride apart, out [m, a + f].
        check(unsafe { (self.k.cat_gelu)(self.g(), att, m as i64, a as i64, mlp, mlp_stride as i64, f as i64, out) }, "concat + gelu")
    }

    /// Attention over short float32 sequences (L <= 4096): scores * scale (+ bias [hq, l, l]), causal or not
    #[allow(clippy::too_many_arguments)]
    pub fn attn_small(&self, q: *const f32, qs: usize, k: *const f32, v: *const f32, kvs: usize, l: usize, hq: usize, hkv: usize, d: usize, scale: f32,
                      bias: *const f32, causal: bool, out: *mut f32) -> Result<()> {
        // SAFETY: q, k, v rows as named, out [l, hq * d], bias [hq, l, l] or null.
        check(unsafe { (self.k.attn_small)(self.g(), q, qs as i64, k, v, kvs as i64, l as i64, hq as i64, hkv as i64, d as i64, scale, bias, causal as c_int, out) },
              "attention")
    }

    /// out [m, n] float32 = x [m, k] half . w^T (w [n, k] half) + bias, accumulated and written in float32
    #[allow(clippy::too_many_arguments)]
    pub fn linear_f32(&self, x: P, m: usize, k: usize, w: P, n: usize, bias: *const f32, out: *mut f32) -> Result<()> {
        // SAFETY: x [m, k], w [n, k], out [m, n], bias [n] or null.
        check(unsafe { (self.k.linear_f32)(self.g(), x, m as i64, k as i64, w, n as i64, bias, out) }, "linear (float32 out)")
    }

    /// x += a y, n floats
    pub fn axpy(&self, x: *mut f32, y: *const f32, n: usize, a: f32) -> Result<()> {
        // SAFETY: n floats each.
        check(unsafe { (self.k.axpy)(self.g(), x, y, n as i64, a) }, "axpy")
    }

    /// half = float32 x s, n values
    pub fn to_half(&self, x: *const f32, out: M, n: usize, s: f32) -> Result<()> {
        // SAFETY: n values each way.
        check(unsafe { (self.k.to_half)(self.g(), x, out, n as i64, s) }, "to half")
    }

    pub fn to_float(&self, x: P, out: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n values each way.
        check(unsafe { (self.k.to_float)(self.g(), x, out, n as i64) }, "to float")
    }

    /// out [n, c] += x [n, ci] repeated c / ci times per channel
    pub fn rep_add(&self, out: M, x: P, n: usize, c: usize, ci: usize) -> Result<()> {
        // SAFETY: out [n, c], x [n, ci].
        check(unsafe { (self.k.rep_add)(self.g(), out, x, n as i64, c as i64, ci as i64, (c / ci) as i64) }, "repeat add")
    }

    /// out [2h, 2w, c] = shuffle(conv [h, w, 4c]) + shuffle(repeat(x [h, w, ci]))
    #[allow(clippy::too_many_arguments)]
    pub fn up_shuffle(&self, conv: P, x: P, h: usize, w: usize, c: usize, ci: usize, out: M) -> Result<()> {
        // SAFETY: conv [h, w, 4c], x [h, w, ci], out [2h, 2w, c].
        check(unsafe { (self.k.up_shuffle)(self.g(), conv, x, h as i64, w as i64, c as i64, ci as i64, (4 * c / ci) as i64, out) }, "upsample")
    }

    /// floats of scratch the group norm needs for g groups (the engine's kernel or the silo's)
    pub fn gn_scratch(&self, g: usize) -> usize {
        // SAFETY: size queries.
        let own = unsafe { (self.k.gn_scratch)(g as i64) as usize };
        self.silo.map_or(own, |s| own.max(unsafe { (s.gn_scratch)(g as i64) } as usize))
    }

    /// GroupNorm (+ SiLU) of x [n, c] half
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm(&self, x: P, n: usize, c: usize, groups: usize, weight: *const f32, bias: *const f32, eps: f32, silu: bool, scratch: *mut f32,
                      out: M) -> Result<()> {
        // SAFETY: x and out [n, c], weight / bias [c], scratch gn_scratch(groups) floats.
        if let Some(s) = self.silo.filter(|s| unsafe { (s.gn_supported)(c as i64, groups as i64) } == 1) {
            return check(unsafe { (s.group_norm)(self.g(), x, n as i64, c as i64, groups as i64, weight, bias, eps, silu as c_int, scratch, out) },
                         "group norm (silo)");
        }
        check(unsafe { (self.k.group_norm)(self.g(), x, n as i64, c as i64, groups as i64, weight, bias, eps, silu as c_int, scratch, out) }, "group norm")
    }

    pub fn transpose(&self, x: P, r: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: x [r, c], out [c, r].
        check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    pub fn softmax_rows(&self, x: *const f32, m: usize, n: usize, scale: f32, out: M) -> Result<()> {
        // SAFETY: x [m, n] float32, out [m, n] half.
        check(unsafe { (self.k.softmax_rows)(self.g(), x, m as i64, n as i64, scale, out) }, "softmax")
    }

    pub fn to_rgb8(&self, x: P, n: usize, out: *mut u8) -> Result<()> {
        // SAFETY: x [n, 3] half, out [n, 3] bytes.
        check(unsafe { (self.k.to_rgb8)(self.g(), x, n as i64, out) }, "to rgb")
    }
}
