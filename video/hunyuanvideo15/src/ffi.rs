//! This engine's own kernels (`kernels/video/hunyuanvideo15/hv15.h`), bound from the video kernel library by name, and
//! its silo (`silo/silo.h`: the DiT's Q8_0 products), with checked wrappers on device pointers (the sizes each names are
//! the caller's to hold).
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
    attn_small: unsafe extern "C" fn(G, *const f32, i64, *const f32, *const f32, i64, i64, i64, i64, i64, f32, *const f32, c_int, *mut f32) -> c_int,
    linear_f32: unsafe extern "C" fn(G, P, i64, i64, P, i64, i64, *const f32, *mut f32, i64) -> c_int,
    linear32: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, i64, *const f32, *mut f32) -> c_int,
    axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    to_half: unsafe extern "C" fn(G, *const f32, M, i64, f32) -> c_int,
    to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    q8_0_float: unsafe extern "C" fn(G, P, i64, *mut f32) -> c_int,
    pad_causal: unsafe extern "C" fn(G, P, P, i64, i64, i64, i64, M) -> c_int,
    rms_channels: unsafe extern "C" fn(G, P, i64, i64, *const f32, c_int, M) -> c_int,
    rep_add: unsafe extern "C" fn(G, M, P, i64, i64, i64, i64) -> c_int,
    add: unsafe extern "C" fn(G, M, P, i64) -> c_int,
    upsample: unsafe extern "C" fn(G, P, P, i64, i64, i64, i64, i64, c_int, c_int, M) -> c_int,
    transpose: unsafe extern "C" fn(G, P, i64, i64, M) -> c_int,
    downsample: unsafe extern "C" fn(G, P, P, i64, i64, i64, i64, i64, M) -> c_int,
    group_mean_add: unsafe extern "C" fn(G, M, P, i64, i64, i64) -> c_int,
    softmax_causal: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, f32, M) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the video library, rebuilt: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in hv15.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            q4_0_half: sym!("ns_video_hv15_q4_0_half"),
            ln_mod: sym!("ns_video_hv15_ln_mod"),
            act: sym!("ns_video_hv15_act"),
            geglu: sym!("ns_video_hv15_geglu"),
            attn_small: sym!("ns_video_hv15_attn_small"),
            linear_f32: sym!("ns_video_hv15_linear_f32"),
            linear32: sym!("ns_video_hv15_linear32"),
            axpy: sym!("ns_video_hv15_axpy"),
            to_half: sym!("ns_video_hv15_to_half"),
            to_float: sym!("ns_video_hv15_to_float"),
            q8_0_float: sym!("ns_video_hv15_q8_0_float"),
            pad_causal: sym!("ns_video_hv15_pad_causal"),
            rms_channels: sym!("ns_video_hv15_rms_channels"),
            rep_add: sym!("ns_video_hv15_rep_add"),
            add: sym!("ns_video_hv15_add"),
            upsample: sym!("ns_video_hv15_upsample_ex"),
            transpose: sym!("ns_video_hv15_transpose"),
            downsample: sym!("ns_video_hv15_downsample"),
            group_mean_add: sym!("ns_video_hv15_group_mean_add"),
            softmax_causal: sym!("ns_video_hv15_softmax_causal"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// The silo's name (`kernels/video/hunyuanvideo15/silo` -> `dist/silo/libnextsycl-hunyuanvideo15.so`)
pub const SILO: &str = "hunyuanvideo15";

/// One product's arguments (silo.h's `ns_hv15_mm`)
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

struct Silo {
    path: std::path::PathBuf,
    q8_split: unsafe extern "C" fn(G, P, i64, i64, *mut i8, M) -> c_int,
    linear_supported: unsafe extern "C" fn(G, c_int) -> c_int,
    linear: unsafe extern "C" fn(G, *const Mm) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

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
            q8_split: sym!("ns_video_hv15_silo_q8_split"),
            linear_supported: sym!("ns_video_hv15_silo_linear_supported"),
            linear: sym!("ns_video_hv15_silo_linear"),
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

/// A buffer's type as the kernels name it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum T {
    F32 = 0,
    F16 = 1,
}

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
    silo: Option<&'static Silo>,
}

impl K {
    pub fn new(gpu: &std::sync::Arc<Gpu>) -> Result<K> {
        let k = api()?;
        // SAFETY: a capability check on the GPU's handle.
        let silo = silo()?.filter(|s| unsafe { (s.linear_supported)(gpu.raw(), 0) } == 1);
        Ok(K { gpu: gpu.clone(), k, silo })
    }

    /// The silo's library, when its products are used here
    pub fn silo(&self) -> Option<&'static std::path::Path> {
        self.silo.map(|s| s.path.as_path())
    }

    fn g(&self) -> G {
        self.gpu.raw()
    }

    /// A Q8_0 matrix [n, k] into the silo's int8 values [n, k] and half scales [k / 32, n]
    pub fn q8_split(&self, blocks: P, n: usize, k: usize, w: M, s: M) -> Result<()> {
        let silo = self.silo.ok_or_else(|| Error("the silo is not loaded".into()))?;
        // SAFETY: blocks holds n x k / 32 Q8_0 blocks, w n x k bytes, s n x k / 32 halves.
        check(unsafe { (silo.q8_split)(self.g(), blocks, n as i64, k as i64, w.cast(), s) }, "q8_0 split")
    }

    /// One of the silo's products (silo.h)
    pub fn mm(&self, a: &Mm) -> Result<()> {
        let silo = self.silo.ok_or_else(|| Error("the silo is not loaded".into()))?;
        // SAFETY: the buffers `a` names hold the rows, columns and strides it gives (the caller's to hold).
        check(unsafe { (silo.linear)(self.g(), a) }, "linear (silo)")
    }

    pub fn q4_0_half(&self, src: P, n: usize, out: M) -> Result<()> {
        // SAFETY: src holds n / 32 blocks, out n halves.
        check(unsafe { (self.k.q4_0_half)(self.g(), src, n as i64, out) }, "q4_0 to half")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ln_mod(&self, x: *const f32, m: usize, c: usize, eps: f32, weight: *const f32, bias: *const f32, scale: *const f32, shift: *const f32,
                  out: M, out_dt: T) -> Result<()> {
        // SAFETY: x [m, c], out [m, c], the vectors [c] or null.
        check(unsafe { (self.k.ln_mod)(self.g(), x, m as i64, c as i64, eps, weight, bias, scale, shift, out, out_dt as c_int) }, "layer norm")
    }

    pub fn act(&self, x: M, dt: T, n: usize, a: Act) -> Result<()> {
        // SAFETY: x holds n values.
        check(unsafe { (self.k.act)(self.g(), x, dt as c_int, n as i64, a as c_int) }, "activation")
    }

    pub fn geglu(&self, x: P, dt: T, m: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: x [m, 2c], out [m, c].
        check(unsafe { (self.k.geglu)(self.g(), x, dt as c_int, m as i64, c as i64, out) }, "gated gelu")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_small(&self, q: *const f32, qs: usize, k: *const f32, v: *const f32, kvs: usize, l: usize, hq: usize, hkv: usize, d: usize, scale: f32,
                      bias: *const f32, causal: bool, out: *mut f32) -> Result<()> {
        // SAFETY: q, k, v rows as named, out [l, hq * d], bias [hq, l, l] or null.
        check(unsafe { (self.k.attn_small)(self.g(), q, qs as i64, k, v, kvs as i64, l as i64, hq as i64, hkv as i64, d as i64, scale, bias, causal as c_int, out) },
              "attention")
    }

    /// out [m, n] (rows ldo apart; 0: n) float32 = x [m, k] half . w^T (w rows ldw apart; 0: k) + bias
    #[allow(clippy::too_many_arguments)]
    pub fn linear_f32(&self, x: P, m: usize, k: usize, w: P, ldw: usize, n: usize, bias: *const f32, out: *mut f32, ldo: usize) -> Result<()> {
        // SAFETY: x [m, k], w n rows of k values ldw apart, out m rows of n values ldo apart, bias [n] or null.
        check(unsafe { (self.k.linear_f32)(self.g(), x, m as i64, k as i64, w, ldw as i64, n as i64, bias, out, ldo as i64) }, "linear (float32 out)")
    }

    /// out [m, n] float32 = x [m, k] float32 . w^T + bias, deterministic
    #[allow(clippy::too_many_arguments)]
    pub fn linear32(&self, x: P, m: usize, k: usize, w: P, n: usize, bias: P, out: M) -> Result<()> {
        // SAFETY: x [m, k], w [n, k], out [m, n] floats, bias [n] or null.
        check(unsafe { (self.k.linear32)(self.g(), x.cast(), m as i64, k as i64, w.cast(), n as i64, bias.cast(), out.cast()) }, "linear (float32)")
    }

    pub fn axpy(&self, x: *mut f32, y: *const f32, n: usize, a: f32) -> Result<()> {
        // SAFETY: n floats each.
        check(unsafe { (self.k.axpy)(self.g(), x, y, n as i64, a) }, "axpy")
    }

    pub fn to_half(&self, x: *const f32, out: M, n: usize, s: f32) -> Result<()> {
        // SAFETY: n values each way.
        check(unsafe { (self.k.to_half)(self.g(), x, out, n as i64, s) }, "to half")
    }

    pub fn to_float(&self, x: P, out: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n values each way.
        check(unsafe { (self.k.to_float)(self.g(), x, out, n as i64) }, "to float")
    }

    pub fn q8_0_float(&self, src: P, n: usize, out: *mut f32) -> Result<()> {
        // SAFETY: src n / 32 blocks, out n floats.
        check(unsafe { (self.k.q8_0_float)(self.g(), src, n as i64, out) }, "q8_0 to float")
    }

    /// x [t, h, w, c] -> out [t + 2, h + 2, w + 2, c] (causal replicate padding; with prev, a padded [2, h + 2, w + 2,
    /// c], those frames first: a stream carried on)
    #[allow(clippy::too_many_arguments)]
    pub fn pad_causal(&self, x: P, prev: P, t: usize, h: usize, w: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: x, prev (or null) and out hold the volumes named.
        check(unsafe { (self.k.pad_causal)(self.g(), x, prev, t as i64, h as i64, w as i64, c as i64, out) }, "causal padding")
    }

    pub fn rms_channels(&self, x: P, n: usize, c: usize, gamma: *const f32, silu: bool, out: M) -> Result<()> {
        // SAFETY: x and out [n, c] half, gamma [c].
        check(unsafe { (self.k.rms_channels)(self.g(), x, n as i64, c as i64, gamma, silu as c_int, out) }, "channel rms norm")
    }

    pub fn rep_add(&self, out: M, x: P, n: usize, c: usize, ci: usize) -> Result<()> {
        // SAFETY: out [n, c], x [n, ci].
        check(unsafe { (self.k.rep_add)(self.g(), out, x, n as i64, c as i64, ci as i64, (c / ci.max(1)) as i64) }, "repeat add")
    }

    pub fn add(&self, x: M, y: P, n: usize) -> Result<()> {
        // SAFETY: n halves each.
        check(unsafe { (self.k.add)(self.g(), x, y, n as i64) }, "add")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsample(&self, conv: P, x: P, t: usize, h: usize, w: usize, c: usize, ci: usize, temporal: bool, first: bool, out: M) -> Result<()> {
        // SAFETY: conv [t, h, w, 8c or 4c], x [t, h, w, ci], out [2t - 1 (2t after the first chunk) or t, 2h, 2w, c].
        check(unsafe { (self.k.upsample)(self.g(), conv, x, t as i64, h as i64, w as i64, c as i64, ci as i64, temporal as c_int, first as c_int, out) },
              "upsample")
    }

    /// conv [h, w, cc] and x [h, w, ci] -> out [h / 2, w / 2, cout] (one frame's Downsample)
    #[allow(clippy::too_many_arguments)]
    pub fn downsample(&self, conv: P, x: P, h: usize, w: usize, cc: usize, ci: usize, cout: usize, out: M) -> Result<()> {
        // SAFETY: conv [h, w, cc], x [h, w, ci], out [h / 2, w / 2, cout] halves.
        check(unsafe { (self.k.downsample)(self.g(), conv, x, h as i64, w as i64, cc as i64, ci as i64, cout as i64, out) }, "downsample")
    }

    /// out [n, c] += the mean of x [n, c r] over runs of r channels
    pub fn group_mean_add(&self, out: M, x: P, n: usize, c: usize, r: usize) -> Result<()> {
        // SAFETY: out [n, c], x [n, c r] halves.
        check(unsafe { (self.k.group_mean_add)(self.g(), out, x, n as i64, c as i64, r as i64) }, "group mean")
    }

    pub fn transpose(&self, x: P, r: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: x [r, c], out [c, r].
        check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn softmax_causal(&self, x: *const f32, m: usize, nk: usize, row0: usize, hw: usize, scale: f32, out: M) -> Result<()> {
        // SAFETY: x [m, nk] float32, out [m, nk] half.
        check(unsafe { (self.k.softmax_causal)(self.g(), x, m as i64, nk as i64, row0 as i64, hw as i64, scale, out) }, "causal softmax")
    }
}

/// The text encoders' kernels (crates/hunyuan-text) are this engine's own (hv15.h)
impl nextsycl_hunyuan_text::Kern for K {
    fn q4_0_half(&self, src: P, n: usize, out: M) -> Result<()> {
        K::q4_0_half(self, src, n, out)
    }
    fn to_half(&self, x: *const f32, out: M, n: usize, s: f32) -> Result<()> {
        K::to_half(self, x, out, n, s)
    }
    fn linear_f32(&self, x: P, m: usize, k: usize, w: P, n: usize, bias: *const f32, out: *mut f32) -> Result<()> {
        K::linear_f32(self, x, m, k, w, 0, n, bias, out, 0)
    }
    fn attn_small(&self, q: *const f32, qs: usize, k: *const f32, v: *const f32, kvs: usize, l: usize, hq: usize, hkv: usize, d: usize, scale: f32,
                  bias: *const f32, causal: bool, out: *mut f32) -> Result<()> {
        K::attn_small(self, q, qs, k, v, kvs, l, hq, hkv, d, scale, bias, causal, out)
    }
    fn geglu_f32(&self, x: P, m: usize, c: usize, out: M) -> Result<()> {
        K::geglu(self, x, T::F32, m, c, out)
    }
}
