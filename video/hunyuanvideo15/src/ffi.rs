//! This engine's own kernels (`kernels/video/hunyuanvideo15/hv15.h`), bound from the video kernel library by name, and
//! its silo (`silo/silo.h`: the DiT's Q8_0 products), with checked wrappers on device pointers (the sizes each names are
//! the caller's to hold).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{DevBuf, Error, Gpu, Result};

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
    kv_q8: unsafe extern "C" fn(G, P, i64, i64, i64, *mut i8, *mut f32) -> c_int,
    kv_dq8: unsafe extern "C" fn(G, *const i8, *const f32, i64, i64, M, i64) -> c_int,
    copy_rows: unsafe extern "C" fn(G, M, i64, P, i64, i64, i64) -> c_int,
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
            kv_q8: sym!("ns_video_hv15_kv_q8"),
            kv_dq8: sym!("ns_video_hv15_kv_dq8"),
            copy_rows: sym!("ns_video_hv15_copy_rows"),
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
    /// the yardstick (silo.h), when this build has it
    bench_gemm: Option<unsafe extern "C" fn(G, c_int, i64, i64, i64, c_int, *mut f64) -> c_int>,
    /// the joint attention (silo.h), when this build has it
    attention: Option<(unsafe extern "C" fn(G) -> c_int, AttnFn)>,
    /// queries and keys apart (silo.h attention2)
    attention2: Option<Attn2Fn>,
}

type Attn2Fn = unsafe extern "C" fn(G, P, i64, i64, P, P, i64, i64, i64, i64, M, i64, c_int, c_int) -> c_int;

type AttnFn = unsafe extern "C" fn(G, P, i64, i64, i64, i64, M, i64, c_int, c_int) -> c_int;

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
            attention: {
                let (t, a) = (lib.symbol("ns_video_hv15_silo_attention_tiles"), lib.symbol("ns_video_hv15_silo_attention"));
                // SAFETY: declared in silo.h with these signatures.
                (!t.is_null() && !a.is_null()).then(|| unsafe {
                    (std::mem::transmute::<*mut c_void, unsafe extern "C" fn(G) -> c_int>(t), std::mem::transmute::<*mut c_void, AttnFn>(a))
                })
            },
            attention2: {
                let p = lib.symbol("ns_video_hv15_silo_attention2");
                // SAFETY: declared in silo.h with this signature.
                (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, Attn2Fn>(p) })
            },
            bench_gemm: {
                let p = lib.symbol("ns_video_hv15_silo_bench_gemm");
                // SAFETY: declared in silo.h with this signature.
                (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn(G, c_int, i64, i64, i64, c_int, *mut f64) -> c_int>(p) })
            },
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

    /// How many half tiles the silo's attention has (0: none - no sycl-tla part, or an older silo)
    pub fn attention_tiles(&self) -> usize {
        // SAFETY: a query on the GPU's handle.
        self.silo.and_then(|s| s.attention).map(|(t, _)| unsafe { t(self.g()) }.max(0) as usize).unwrap_or(0)
    }

    /// The joint attention: qkv half [s, stride] (q | k | v, H heads of 128 each), out half [s, o_rs]; sage: int8 q, k
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, qkv: P, s: usize, h: usize, d: usize, stride: usize, out: M, o_rs: usize, sage: bool, tile: usize) -> Result<()> {
        let (_, f) = self.silo.and_then(|s| s.attention).ok_or_else(|| Error("the silo has no attention (rebuild it)".into()))?;
        // SAFETY: qkv holds s rows of stride halves, out s rows of o_rs.
        check(unsafe { f(self.g(), qkv, s as i64, h as i64, d as i64, stride as i64, out, o_rs as i64, sage as c_int, tile as c_int) }, "attention (silo)")
    }

    /// Whether the silo has the attention over queries and keys apart
    pub fn has_attention2(&self) -> bool {
        self.silo.and_then(|s| s.attention2).is_some() && self.attention_tiles() > 0
    }

    /// Attention of `sq` query rows (q, `qs` halves apart) over `skv` key / value rows (k, v, `kvs` apart): out half
    /// rows `o_rs` apart; sage: int8 q, k
    #[allow(clippy::too_many_arguments)]
    pub fn attention2(&self, q: P, qs: usize, sq: usize, k: P, v: P, kvs: usize, skv: usize, h: usize, d: usize, out: M, o_rs: usize, sage: bool, tile: usize)
                      -> Result<()> {
        let f = self.silo.and_then(|s| s.attention2).ok_or_else(|| Error("the silo has no attention2 (rebuild it)".into()))?;
        // SAFETY: q holds sq rows of qs halves, k and v skv rows of kvs, out sq rows of o_rs.
        check(unsafe { f(self.g(), q, qs as i64, sq as i64, k, v, kvs as i64, skv as i64, h as i64, d as i64, out, o_rs as i64, sage as c_int, tile as c_int) },
              "attention (silo, apart)")
    }

    /// oneDNN's plain product [m, k] x [k, n] (int8: int8 -> int32, else half -> half): milliseconds a run, mean of `iters`
    pub fn bench_gemm(&self, int8: bool, m: usize, n: usize, k: usize, iters: usize) -> Result<f64> {
        let f = self.silo.and_then(|s| s.bench_gemm).ok_or_else(|| Error("the silo has no bench_gemm (rebuild it)".into()))?;
        let mut ms = 0f64;
        // SAFETY: the silo allocates its own buffers; ms is written once.
        check(unsafe { f(self.g(), int8 as c_int, m as i64, n as i64, k as i64, iters as c_int, &mut ms) }, "bench_gemm")?;
        Ok(ms)
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

    /// rows of `heads` heads of 128 halves (`src_rs` apart) -> int8 [rows, heads x 128] + a float32 scale a (row, head)
    pub fn kv_q8(&self, src: P, rows: usize, heads: usize, src_rs: usize, dst: M, scale: M) -> Result<()> {
        // SAFETY: src rows x src_rs halves, dst rows x heads x 128 bytes, scale rows x heads floats.
        check(unsafe { (self.k.kv_q8)(self.g(), src, rows as i64, heads as i64, src_rs as i64, dst.cast(), scale.cast()) }, "kv int8")
    }

    /// The other way: int8 + scales -> half rows `dst_rs` apart
    pub fn kv_dq8(&self, src: P, scale: P, rows: usize, heads: usize, dst: M, dst_rs: usize) -> Result<()> {
        // SAFETY: as kv_q8's.
        check(unsafe { (self.k.kv_dq8)(self.g(), src.cast(), scale.cast(), rows as i64, heads as i64, dst, dst_rs as i64) }, "kv int8 -> half")
    }

    /// `height` rows of `width` bytes, `src` + `src_at` (rows `src_pitch` apart) to `dst` + `at` (`dst_pitch` apart), both
    /// buffers' bounds checked: DevBuf::copy_2d's job, with every work-item 16 bytes instead of a row
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows(&self, dst: &DevBuf, at: usize, dst_pitch: usize, src: &DevBuf, src_at: usize, src_pitch: usize, width: usize, height: usize) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        let (dn, sn) = ((height - 1) * dst_pitch + width, (height - 1) * src_pitch + width);
        if at + dn > dst.len || src_at + sn > src.len {
            return Err(Error(format!("copy_rows: {height} rows of {width} bytes past a buffer ({at} + {dn} of {}, {src_at} + {sn} of {})", dst.len, src.len)));
        }
        // SAFETY: both ranges inside their buffers (checked), on this GPU.
        check(unsafe {
            (self.k.copy_rows)(self.g(), dst.ptr().cast::<u8>().wrapping_add(at).cast(), dst_pitch as i64, src.ptr().cast::<u8>().wrapping_add(src_at).cast_const().cast(),
                               src_pitch as i64, width as i64, height as i64)
        }, "copying rows")
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
