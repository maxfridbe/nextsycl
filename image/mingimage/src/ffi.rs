//! This engine's own kernels (`kernels/image/mingimage/mi.h`), bound from the image kernel library by name.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_int, c_void};
use std::sync::{Arc, Mutex, OnceLock};

use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{Dt, Nsd};

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub ln_mod: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const f32, M) -> c_int,
    pub rms_mod: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, f32, *const f32, M) -> c_int,
    pub softmax_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M) -> c_int,
    pub silu: unsafe extern "C" fn(G, M, i64) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    pub to_rgba8: unsafe extern "C" fn(G, P, i64, i64, *mut u8) -> c_int,
    pub gather_rows: unsafe extern "C" fn(G, P, i64, *const i32, i64, M) -> c_int,
    pub combine: unsafe extern "C" fn(G, *mut f32, i64, i64, P, *const i32, *const f32, i64) -> c_int,
    pub gate_add: unsafe extern "C" fn(G, *mut f32, i64, i64, P, *const f32) -> c_int,
    pub swiglu_scaled: unsafe extern "C" fn(G, P, i64, i64, f32, M) -> c_int,
    pub to_half_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M, i64) -> c_int,
    pub repeat_heads: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, M) -> c_int,
    pub copy_2d: unsafe extern "C" fn(G, M, i64, P, i64, i64, i64) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api() -> nextsycl_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (the image library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in mi.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            ln_mod: sym!("ns_image_mi_ln_mod"),
            rms_mod: sym!("ns_image_mi_rms_mod"),
            softmax_rows: sym!("ns_image_mi_softmax_rows"),
            silu: sym!("ns_image_mi_silu"),
            axpy: sym!("ns_image_mi_axpy"),
            to_half: sym!("ns_image_mi_to_half"),
            to_float: sym!("ns_image_mi_to_float"),
            up2: sym!("ns_image_mi_up2"),
            to_rgba8: sym!("ns_image_mi_to_rgba8"),
            gather_rows: sym!("ns_image_mi_gather_rows"),
            combine: sym!("ns_image_mi_combine"),
            gate_add: sym!("ns_image_mi_gate_add"),
            swiglu_scaled: sym!("ns_image_mi_swiglu_scaled"),
            to_half_rows: sym!("ns_image_mi_to_half_rows"),
            repeat_heads: sym!("ns_image_mi_repeat_heads"),
            copy_2d: sym!("ns_image_mi_copy_2d"),
        })
    })
    .as_ref()
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The scale the MLPs' gated products are made at in half (their next linear's float32 output multiplied back)
pub const ACT_SCALE: f32 = 1.0 / 256.0;

/// out [m, c] half = silu(x[:, :c]) * x[:, c:] * ACT_SCALE (x [m, 2c] half)
pub fn swiglu(gpu: &nextsycl_core::Gpu, x: P, m: usize, c: usize, out: M) -> nextsycl_core::Result<()> {
    let k = api()?;
    // SAFETY: x holds m x 2c halves, out m x c.
    check(unsafe { (k.swiglu_scaled)(gpu.raw(), x, m as i64, c as i64, ACT_SCALE, out) }, "swiglu")
}

/// `height` rows of `width` bytes from `src` (`src_pitch` apart) to `dst` (`dst_pitch` apart), on the GPU's queue
pub fn copy_2d(gpu: &nextsycl_core::Gpu, dst: M, dst_pitch: usize, src: P, src_pitch: usize, width: usize, height: usize) -> nextsycl_core::Result<()> {
    let k = api()?;
    // SAFETY: the caller's buffers hold the rows named.
    check(unsafe { (k.copy_2d)(gpu.raw(), dst, dst_pitch as i64, src, src_pitch as i64, width as i64, height as i64) }, "copy rows")
}

/// Bytes of one row of `k` values of GGUF type `qtype` (F32, F16, Q8_0, Q4_K, Q6_K, BF16: the types this model's files hold)
fn row_bytes(qtype: u32, k: usize) -> Option<usize> {
    match qtype {
        0 => Some(k * 4),
        1 | 30 => Some(k * 2),
        8 if k.is_multiple_of(32) => Some(k / 32 * 34),
        12 if k.is_multiple_of(256) => Some(k / 256 * 144),
        14 if k.is_multiple_of(256) => Some(k / 256 * 210),
        _ => None,
    }
}

/// The weight slices and staged outputs of `linear_q`, a GPU's (they grow to the largest asked; kept for the process)
struct Scratch {
    gpu: usize,
    wt: Option<DevBuf>,
    tmp: Option<DevBuf>,
}

static SCRATCH: Mutex<Vec<Scratch>> = Mutex::new(Vec::new());

fn grown(slot: &mut Option<DevBuf>, gpu: &Arc<nextsycl_core::Gpu>, bytes: usize) -> nextsycl_core::Result<*mut c_void> {
    if slot.as_ref().is_none_or(|b| b.len < bytes) {
        // the old one's work is queued before its memory goes back
        gpu.sync()?;
        *slot = None;
        *slot = Some(DevBuf::new(gpu, bytes)?);
    }
    Ok(slot.as_ref().map(|b| b.ptr()).unwrap_or(std::ptr::null_mut()))
}

/// out [m, n] (rows `ldo` apart, 0: n) = x [m, k] half . W^T + bias, W a GGUF matrix [n, k] of type `qtype` kept as
/// stored on the card: a slice of its rows (about 64 MiB) made half at a time by the shared dequant, multiplied by the
/// shared product (nextcuda's nsd_linear_q, composed here from nextsycl's kernels); `bias` float32 [n] or null
#[allow(clippy::too_many_arguments)]
pub fn linear_q(nsd: &Nsd, x: P, dt: Dt, m: usize, k: usize, w: P, qtype: u32, n: usize, bias: P, out: M, out_dt: Dt, ldo: usize) -> nextsycl_core::Result<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    if dt != Dt::F16 {
        return Err(nextsycl_core::Error("linear_q: activations in half".into()));
    }
    let rb = row_bytes(qtype, k).ok_or_else(|| nextsycl_core::Error(format!("linear_q: a row of {k} values is not whole blocks of GGUF type {qtype}")))?;
    let gpu = &nsd.gpu;
    let ldo = if ldo == 0 { n } else { ldo };
    let oe = if out_dt == Dt::F32 { 4 } else { 2 };
    let rows = ((64usize << 20) / (k * 2)).clamp(1, n);
    let mut sc = SCRATCH.lock().unwrap_or_else(|p| p.into_inner());
    let i = match sc.iter().position(|s| s.gpu == gpu.index) {
        Some(i) => i,
        None => {
            sc.push(Scratch { gpu: gpu.index, wt: None, tmp: None });
            sc.len() - 1
        }
    };
    let s = &mut sc[i];
    let wt = if qtype == 1 { std::ptr::null_mut() } else { grown(&mut s.wt, gpu, rows * k * 2)? };
    // one slice writing straight into a dense output; otherwise each slice staged, then put in its columns
    let direct = rows == n && ldo == n;
    let tmp = if direct { std::ptr::null_mut() } else { grown(&mut s.tmp, gpu, m * rows * oe)? };
    let mut r0 = 0;
    while r0 < n {
        let nn = rows.min(n - r0);
        let src: P = w.cast::<u8>().wrapping_add(r0 * rb).cast();
        let ws: P = match qtype {
            1 => src,
            0 => {
                let api = api()?;
                // SAFETY: src holds nn x k floats, wt nn x k halves.
                check(unsafe { (api.to_half)(gpu.raw(), src.cast(), wt, (nn * k) as i64) }, "to half")?;
                wt
            }
            q => {
                nsd.dequant(src, q, nn * k, wt, Dt::F16)?;
                wt
            }
        };
        let b: P = if bias.is_null() { bias } else { bias.cast::<f32>().wrapping_add(r0).cast() };
        if direct {
            nsd.linear(x, Dt::F16, m, k, ws, nn, b, out, out_dt)?;
        } else {
            nsd.linear(x, Dt::F16, m, k, ws, nn, b, tmp, out_dt)?;
            copy_2d(gpu, out.cast::<u8>().wrapping_add(r0 * oe).cast(), ldo * oe, tmp, nn * oe, nn * oe, m)?;
        }
        r0 += nn;
    }
    Ok(())
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextsycl_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextsycl_core::Error(format!("{what}: {why}")))
}
