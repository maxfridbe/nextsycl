//! The shared diffusion kernels (`kernels/diffusion/nsd.h`: H3's, linked into the image and video kernel libraries),
//! bound by name. `Nsd` is one context on one GPU - around that GPU's own queue, so the buffers nextsycl-core allocates
//! there are the kernels' to read - with checked calls on raw device pointers (`DevBuf::ptr`).
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names (each call's doc says them).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::{Arc, OnceLock};

use nextsycl_core::{Error, Gpu, Result};

/// A buffer's element type, as the kernels name it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Dt {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

type P = *const c_void;
type M = *mut c_void;

#[allow(clippy::type_complexity)]
struct Api {
    last_error: unsafe extern "C" fn() -> *const c_char,
    create: unsafe extern "C" fn(M) -> M,
    destroy: unsafe extern "C" fn(M),
    wait: unsafe extern "C" fn(M) -> c_int,
    int8_linear: unsafe extern "C" fn(M, P, c_int, i64, i64, *const i8, i64, *const f32, i64, *const f32, M, c_int, c_int) -> c_int,
    linear: unsafe extern "C" fn(M, P, c_int, i64, i64, P, i64, *const f32, M, c_int) -> c_int,
    linear_acc: unsafe extern "C" fn(M, P, c_int, i64, i64, P, i64, *const f32, M) -> c_int,
    dequant: unsafe extern "C" fn(M, P, c_int, i64, M, c_int) -> c_int,
    rms_norm_mod: unsafe extern "C" fn(M, P, c_int, i64, i64, *const f32, f32, *const i32, *const f32, *const f32, M, c_int) -> c_int,
    layer_norm: unsafe extern "C" fn(M, P, c_int, i64, i64, *const f32, *const f32, f32, M, c_int) -> c_int,
    rms_rope: unsafe extern "C" fn(M, M, c_int, i64, i64, i64, i64, *const f32, f32, *const f32, c_int) -> c_int,
    swiglu: unsafe extern "C" fn(M, P, c_int, i64, i64, M, c_int) -> c_int,
    gate_add: unsafe extern "C" fn(M, M, c_int, i64, i64, P, c_int, *const i32, *const f32) -> c_int,
    attention: unsafe extern "C" fn(M, P, P, P, c_int, i64, i64, i64, i64, M, c_int) -> c_int,
    attention_causal: unsafe extern "C" fn(M, P, P, P, c_int, i64, i64, i64, i64, i64, i64, M) -> c_int,
    attention_qk: unsafe extern "C" fn(M, P, i64, i64, P, P, i64, i64, c_int, i64, i64, M, c_int) -> c_int,
    conv2d: unsafe extern "C" fn(M, P, c_int, i64, i64, i64, i64, P, i64, i64, *const f32, M) -> c_int,
    attention_batch: unsafe extern "C" fn(M, P, P, P, c_int, i64, i64, i64, i64, i64, M, c_int) -> c_int,
    conv1d: unsafe extern "C" fn(M, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, i64, *mut f32, i64) -> c_int,
    conv_transpose1d: unsafe extern "C" fn(M, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, *mut f32, i64) -> c_int,
    snake: unsafe extern "C" fn(M, *const f32, i64, i64, i64, *const f32, *mut f32) -> c_int,
    scale: unsafe extern "C" fn(M, *mut f32, i64, f32) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (an image or video library, built with kernels/diffusion)", $name));
                }
                // SAFETY: the symbol is declared in nsd.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            last_error: sym!("nsd_last_error"),
            create: sym!("nsd_create"),
            destroy: sym!("nsd_destroy"),
            wait: sym!("nsd_wait"),
            int8_linear: sym!("nsd_int8_linear"),
            linear: sym!("nsd_linear"),
            linear_acc: sym!("nsd_linear_acc"),
            dequant: sym!("nsd_dequant"),
            rms_norm_mod: sym!("nsd_rms_norm_mod"),
            layer_norm: sym!("nsd_layer_norm"),
            rms_rope: sym!("nsd_rms_rope"),
            swiglu: sym!("nsd_swiglu"),
            gate_add: sym!("nsd_gate_add"),
            attention: sym!("nsd_attention"),
            attention_causal: sym!("nsd_attention_causal"),
            attention_qk: sym!("nsd_attention_qk"),
            conv2d: sym!("nsd_conv2d"),
            attention_batch: sym!("nsd_attention_batch"),
            conv1d: sym!("nsd_conv1d"),
            conv_transpose1d: sym!("nsd_conv_transpose1d"),
            snake: sym!("nsd_snake"),
            scale: sym!("nsd_scale"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// The kernels' context on one GPU
pub struct Nsd {
    ctx: M,
    k: &'static Api,
    pub gpu: Arc<Gpu>,
}

// SAFETY: the context is used through &self from one thread at a time by its engine (the queue is in order).
unsafe impl Send for Nsd {}
unsafe impl Sync for Nsd {}

impl Drop for Nsd {
    fn drop(&mut self) {
        // SAFETY: made by nsd_create, destroyed once.
        unsafe { (self.k.destroy)(self.ctx) };
    }
}

impl Nsd {
    /// A context on `gpu`'s own queue
    pub fn new(gpu: &Arc<Gpu>) -> Result<Nsd> {
        let k = api()?;
        // SAFETY: the GPU's queue, alive as long as the GPU (held here).
        let ctx = unsafe { (k.create)(gpu.queue()) };
        if ctx.is_null() {
            return Err(Error(format!("the diffusion kernels' context on {}: {}", gpu.name, last(k))));
        }
        Ok(Nsd { ctx, k, gpu: gpu.clone() })
    }

    fn ok(&self, rc: c_int, what: &str) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(Error(format!("{what}: {}", last(self.k))))
        }
    }

    pub fn wait(&self) -> Result<()> {
        // SAFETY: a live context.
        self.ok(unsafe { (self.k.wait)(self.ctx) }, "waiting for the GPU")
    }

    /// out [M, N] = x [M, K] . W^T with int8 W [N, K] (ComfyUI's int8, ConvRot when `group` > 0: activations rotated
    /// a group at a time, quantized per row), `wscale` per output (n_wscale N) or one; bias float32 [N] or null
    #[allow(clippy::too_many_arguments)]
    pub fn int8_linear(&self, x: P, x_dt: Dt, m: usize, k: usize, w: P, n: usize, wscale: P, n_wscale: usize, bias: P, out: M, out_dt: Dt,
                       group: usize) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.int8_linear)(self.ctx, x, x_dt as c_int, m as i64, k as i64, w.cast(), n as i64, wscale.cast(), n_wscale as i64, bias.cast(), out,
                                               out_dt as c_int, group as c_int) }, "int8 linear")
    }

    /// out [M, N] = x [M, K] . W^T + bias, x and W in `dt`
    #[allow(clippy::too_many_arguments)]
    pub fn linear(&self, x: P, dt: Dt, m: usize, k: usize, w: P, n: usize, bias: P, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.linear)(self.ctx, x, dt as c_int, m as i64, k as i64, w, n as i64, bias.cast(), out, out_dt as c_int) }, "linear")
    }

    /// out [M, N] += x [M, K] . W^T, all in `dt` (a LoRA's update merged into a matrix: x = B, W = A^T)
    #[allow(clippy::too_many_arguments)]
    pub fn linear_acc(&self, x: P, dt: Dt, m: usize, k: usize, w: P, n: usize, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.linear_acc)(self.ctx, x, dt as c_int, m as i64, k as i64, w, n as i64, std::ptr::null(), out) }, "linear (accumulate)")
    }

    /// `n` values of GGUF type `qtype` (8 Q8_0, 12 Q4_K, 14 Q6_K, 30 BF16) into `out_dt`
    pub fn dequant(&self, src: P, qtype: u32, n: usize, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: src holds n values of the type, out n values.
        self.ok(unsafe { (self.k.dequant)(self.ctx, src, qtype as c_int, n as i64, out, out_dt as c_int) }, "dequant")
    }

    /// RMS norm a row (weight float32 [C] or null), then * (1 + scale[rows[r]]) + shift[rows[r]] when given
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_mod(&self, x: P, x_dt: Dt, m: usize, c: usize, weight: P, eps: f32, rows: P, scale: P, shift: P, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.rms_norm_mod)(self.ctx, x, x_dt as c_int, m as i64, c as i64, weight.cast(), eps, rows.cast(), scale.cast(), shift.cast(), out,
                                                out_dt as c_int) }, "rms norm")
    }

    /// Layer norm a row; weight / bias float32 [C] or null
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(&self, x: P, x_dt: Dt, m: usize, c: usize, weight: P, bias: P, eps: f32, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.layer_norm)(self.ctx, x, x_dt as c_int, m as i64, c as i64, weight.cast(), bias.cast(), eps, out, out_dt as c_int) }, "layer norm")
    }

    /// Per-head RMS norm (weight [D]) then RoPE on pairs (i, rot/2 + i) with (cos, sin) per token and pair `cs`, in place
    #[allow(clippy::too_many_arguments)]
    pub fn rms_rope(&self, x: M, dt: Dt, m: usize, h: usize, d: usize, stride: usize, weight: P, eps: f32, cs: P, rot: usize) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.rms_rope)(self.ctx, x, dt as c_int, m as i64, h as i64, d as i64, stride as i64, weight.cast(), eps, cs.cast(), rot as c_int) }, "rms + rope")
    }

    /// out [M, C] = silu(x[:, :C]) * x[:, C:]
    pub fn swiglu(&self, x: P, x_dt: Dt, m: usize, c: usize, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: x holds [M, 2C], out [M, C].
        self.ok(unsafe { (self.k.swiglu)(self.ctx, x, x_dt as c_int, m as i64, c as i64, out, out_dt as c_int) }, "swiglu")
    }

    /// x[r] += other[r] * gate[rows[r]] (gate float32 [R, C]; rows / gate null: a plain add)
    #[allow(clippy::too_many_arguments)]
    pub fn gate_add(&self, x: M, x_dt: Dt, m: usize, c: usize, other: P, other_dt: Dt, rows: P, gate: P) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.gate_add)(self.ctx, x, x_dt as c_int, m as i64, c as i64, other, other_dt as c_int, rows.cast(), gate.cast()) }, "gate add")
    }

    /// Full attention over S rows of H heads x D (rows `stride` apart), out [S, H * D]
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, q: P, k: P, v: P, dt: Dt, s: usize, h: usize, d: usize, stride: usize, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.attention)(self.ctx, q, k, v, dt as c_int, s as i64, h as i64, d as i64, stride as i64, out, out_dt as c_int) }, "attention")
    }

    /// Causal attention, Hq query heads over Hkv key/value heads; q rows qs apart, k / v kvs apart; out [L, Hq * D] in dt
    #[allow(clippy::too_many_arguments)]
    pub fn attention_causal(&self, q: P, k: P, v: P, dt: Dt, l: usize, hq: usize, hkv: usize, d: usize, qs: usize, kvs: usize, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.attention_causal)(self.ctx, q, k, v, dt as c_int, l as i64, hq as i64, hkv as i64, d as i64, qs as i64, kvs as i64, out) },
                "causal attention")
    }

    /// Sq query rows over Skv key / value rows, no mask; out [Sq, H * D]
    #[allow(clippy::too_many_arguments)]
    pub fn attention_qk(&self, q: P, sq: usize, qs: usize, k: P, v: P, skv: usize, kvs: usize, dt: Dt, h: usize, d: usize, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.attention_qk)(self.ctx, q, sq as i64, qs as i64, k, v, skv as i64, kvs as i64, dt as c_int, h as i64, d as i64, out, out_dt as c_int) },
                "attention (prefix + block)")
    }

    /// A k x k convolution, stride 1, same padding: x [N, H, W, Ci] -> out [N, H, W, Co] in `dt`; w [Co, Ci, k, k] in `dt`
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(&self, x: P, dt: Dt, n: usize, h: usize, w_: usize, ci: usize, w: P, co: usize, k: usize, bias: P, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.conv2d)(self.ctx, x, dt as c_int, n as i64, h as i64, w_ as i64, ci as i64, w, co as i64, k as i64, bias.cast(), out) }, "conv2d")
    }

    /// B independent sequences of S rows each (sequence b's rows from row b * S of q, k, v and out), full attention
    #[allow(clippy::too_many_arguments)]
    pub fn attention_batch(&self, q: P, k: P, v: P, dt: Dt, b: usize, s: usize, h: usize, d: usize, stride: usize, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.attention_batch)(self.ctx, q, k, v, dt as c_int, b as i64, s as i64, h as i64, d as i64, stride as i64, out, out_dt as c_int) },
                "attention (batch)")
    }

    /// A 1-D convolution of float32 signals x [B, Ci, L] with w [Co, Ci, K] (bias [Co] or null), zero padding `pad`
    /// both sides: out [B, Co, Lo], Lo = (L + 2 pad - dil (K - 1) - 1) / stride + 1
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(&self, x: P, b: usize, ci: usize, l: usize, w: P, co: usize, k: usize, bias: P, stride: usize, dil: usize, pad: usize, out: M,
                  lo: usize) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.conv1d)(self.ctx, x.cast(), b as i64, ci as i64, l as i64, w.cast(), co as i64, k as i64, bias.cast(), stride as i64, dil as i64,
                                          pad as i64, out.cast(), lo as i64) }, "conv1d")
    }

    /// The transposed 1-D convolution: x [B, Ci, L], w [Ci, Co, K]; out [B, Co, Lo], Lo = (L - 1) stride - 2 pad + K
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(&self, x: P, b: usize, ci: usize, l: usize, w: P, co: usize, k: usize, bias: P, stride: usize, pad: usize, out: M,
                            lo: usize) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.conv_transpose1d)(self.ctx, x.cast(), b as i64, ci as i64, l as i64, w.cast(), co as i64, k as i64, bias.cast(),
                                                    stride as i64, pad as i64, out.cast(), lo as i64) }, "conv_transpose1d")
    }

    /// Snake: x + sin^2(alpha x) / alpha per channel of x [B, C, L] float32 (alpha [C])
    #[allow(clippy::too_many_arguments)]
    pub fn snake(&self, x: P, b: usize, c: usize, l: usize, alpha: P, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(unsafe { (self.k.snake)(self.ctx, x.cast(), b as i64, c as i64, l as i64, alpha.cast(), out.cast()) }, "snake")
    }

    /// x *= s, float32
    pub fn scale(&self, x: M, n: usize, s: f32) -> Result<()> {
        // SAFETY: x holds n floats.
        self.ok(unsafe { (self.k.scale)(self.ctx, x.cast(), n as i64, s) }, "scale")
    }
}

fn last(k: &Api) -> String {
    // SAFETY: a C string owned by the library, valid until its next call on this thread.
    unsafe { CStr::from_ptr((k.last_error)()) }.to_string_lossy().into_owned()
}

/// A null pointer (an absent optional argument)
pub fn none() -> P {
    std::ptr::null()
}
