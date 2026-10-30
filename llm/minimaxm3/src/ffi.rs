//! The kernels this engine runs, bound by name from the language kind's kernel library: its own
//! (`kernels/llm/minimaxm3/mm3.h`) and the generic ones of the glm5next engine's (`kernels/llm/glm5next/glm.h`: the
//! dequantization, the GEMMs, the RMS norm, the gathers).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type Gpu = *mut c_void;

pub struct Api {
    // glm.h
    pub dequant: unsafe extern "C" fn(Gpu, c_int, *const c_void, usize, *mut f32) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(Gpu, c_int, *const c_void, i64, *mut u16) -> c_int,
    pub gemm: unsafe extern "C" fn(Gpu, i64, i64, i64, *const f32, i64, *const f32, *mut f32, i64, c_int) -> c_int,
    pub gemm_f16: unsafe extern "C" fn(Gpu, i64, i64, i64, *const u16, i64, *const u16, *mut f32, i64, c_int) -> c_int,
    pub rms_norm: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub to_f16: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64) -> c_int,
    pub gather_f16: unsafe extern "C" fn(Gpu, *const f32, *const i32, *mut u16, i64, i64) -> c_int,
    pub add: unsafe extern "C" fn(Gpu, *mut f32, *const f32, i64) -> c_int,
    pub gather: unsafe extern "C" fn(Gpu, *const f32, *const i32, *mut f32, i64, i64) -> c_int,
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub quantize_q8_1: unsafe extern "C" fn(Gpu, *const f32, *mut c_void, i64, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(Gpu, c_int, *const c_void, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    // mm3.h
    pub rope: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, i64, i64, i64, i64, f32) -> c_int,
    pub scatter_rank: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const i32, *const f32, *const i32, i64, i64, i64) -> c_int,
    pub swiglu_oai: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, f32, f32) -> c_int,
    pub idx_scores: unsafe extern "C" fn(Gpu, *const f32, *const u16, *mut f32, i64, i64, i64, i64, i64) -> c_int,
    pub idx_topk: unsafe extern "C" fn(Gpu, *mut f32, *mut i32, i64, i64, i64, i64) -> c_int,
    pub attend: AttendFn,
    /// the silo's prompt-pass attention (kernels/llm/minimaxm3/silo/mm3s.h), when the silo is there
    pub silo_attend: Option<AttendFn>,
    /// the silo's decode attention (split over the keys; takes the split count and a scratch for the partial states)
    pub silo_attend_split: Option<SplitFn>,
    /// the silo's expert products for a decode token (moe.cpp)
    pub silo_moe: Option<MoeFns>,
    /// the silo's fused steps of a decode token (moe.cpp)
    pub silo_one: Option<OneFns>,
    /// the silo's decode products on Q8_1 activations (mmvq.cpp)
    pub silo_q: Option<QFns>,
}

pub struct QFns {
    pub supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    pub split_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub quantize: unsafe extern "C" fn(Gpu, *const f32, *mut c_void, *mut c_void, i64, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(Gpu, c_int, *const c_void, *const c_void, *const c_void, *mut f32, i64, i64, i64) -> c_int,
}

pub struct OneFns {
    pub mv3: unsafe extern "C" fn(Gpu, c_int, *const c_void, *const c_void, *const c_void, *const f32, *mut f32, *mut f32, *mut f32, i64, i64, i64, i64) -> c_int,
    pub post: PostFn,
    pub add_norm: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const f32, *const f32, *mut f32, i64, f32) -> c_int,
    pub finish: unsafe extern "C" fn(Gpu, *const f32, i64, *const f32, *const f32, *mut f32, i64) -> c_int,
    pub to_half: unsafe extern "C" fn(Gpu, c_int, *const c_void, i64, *mut u16) -> c_int,
}

pub type PostFn = unsafe extern "C" fn(Gpu, *mut f32, *mut f32, *const f32, *mut f32, *mut f32, *const f32, *const f32, *const f32, *const f32, *mut u16, *mut u16, *mut u16, i64, i64, i64, i64, i64, i64, f32, f32) -> c_int;

pub struct MoeFns {
    pub supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    pub gu: unsafe extern "C" fn(Gpu, c_int, *const u64, i64, *const f32, *mut f32, i64, i64, i64, f32, f32) -> c_int,
    pub down: unsafe extern "C" fn(Gpu, c_int, *const u64, i64, *const f32, *const f32, *const i32, *mut f32, i64, i64, i64, i64) -> c_int,
}

pub type SplitFn = unsafe extern "C" fn(Gpu, *const f32, *const u16, *const u16, *const i32, i64, *mut f32, i64, i64, i64, i64, f32, i64, *mut f32) -> c_int;

pub type AttendFn = unsafe extern "C" fn(Gpu, *const f32, *const u16, *const u16, *const i32, i64, *mut f32, i64, i64, i64, i64, f32) -> c_int;

/// The table, bound once (an error when the library lacks a symbol: an older build)
pub fn api() -> nextsycl_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (rebuild it: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in glm.h / mm3.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            dequant: sym!("ns_dequant"),
            dequant_f16: sym!("ns_dequant_f16"),
            gemm: sym!("ns_gemm"),
            gemm_f16: sym!("ns_gemm_f16"),
            rms_norm: sym!("ns_rms_norm"),
            to_f16: sym!("ns_to_f16"),
            gather_f16: sym!("ns_gather_f16"),
            add: sym!("ns_add"),
            gather: sym!("ns_gather"),
            q8_1_bytes: sym!("ns_q8_1_bytes"),
            mmvq_supported: sym!("ns_mmvq_supported"),
            quantize_q8_1: sym!("ns_quantize_q8_1"),
            mmvq: sym!("ns_mmvq"),
            rope: sym!("ns_llm_minimaxm3_rope"),
            scatter_rank: sym!("ns_llm_minimaxm3_scatter_rank"),
            swiglu_oai: sym!("ns_llm_minimaxm3_swiglu_oai"),
            idx_scores: sym!("ns_llm_minimaxm3_idx_scores"),
            idx_topk: sym!("ns_llm_minimaxm3_idx_topk"),
            attend: sym!("ns_llm_minimaxm3_attend"),
            silo_moe: nextsycl_core::silo(crate::ARCH).ok().flatten().and_then(|s| {
                let (a, b, c) = (s.symbol("ns_mm3s_expert_supported"), s.symbol("ns_mm3s_expert_gu"), s.symbol("ns_mm3s_expert_down"));
                // SAFETY: the symbols are declared in mm3s.h with MoeFns's signatures.
                #[allow(clippy::missing_transmute_annotations)]
                (!a.is_null() && !b.is_null() && !c.is_null()).then(|| unsafe {
                    MoeFns {
                        supported: std::mem::transmute::<*mut c_void, _>(a),
                        gu: std::mem::transmute::<*mut c_void, _>(b),
                        down: std::mem::transmute::<*mut c_void, _>(c),
                    }
                })
            }),
            silo_q: nextsycl_core::silo(crate::ARCH).ok().flatten().and_then(|s| {
                let (a, b, c, d) = (s.symbol("ns_mm3q_mmvq_supported"), s.symbol("ns_mm3q_q8_split_bytes"), s.symbol("ns_mm3q_quantize_q8"), s.symbol("ns_mm3q_mmvq"));
                // SAFETY: the symbols are declared in mm3q.h with QFns's signatures.
                #[allow(clippy::missing_transmute_annotations)]
                (!a.is_null() && !b.is_null() && !c.is_null() && !d.is_null()).then(|| unsafe {
                    QFns {
                        supported: std::mem::transmute::<*mut c_void, _>(a),
                        split_bytes: std::mem::transmute::<*mut c_void, _>(b),
                        quantize: std::mem::transmute::<*mut c_void, _>(c),
                        mmvq: std::mem::transmute::<*mut c_void, _>(d),
                    }
                })
            }),
            silo_one: nextsycl_core::silo(crate::ARCH).ok().flatten().and_then(|s| {
                let (a, b, c, d) = (s.symbol("ns_mm3s_mv3"), s.symbol("ns_mm3s_attn_post"), s.symbol("ns_mm3s_add_norm"), s.symbol("ns_mm3s_finish"));
                let e = s.symbol("ns_mm3s_to_half");
                // SAFETY: the symbols are declared in mm3s.h with OneFns's signatures.
                #[allow(clippy::missing_transmute_annotations)]
                (!a.is_null() && !b.is_null() && !c.is_null() && !d.is_null() && !e.is_null()).then(|| unsafe {
                    OneFns {
                        mv3: std::mem::transmute::<*mut c_void, _>(a),
                        post: std::mem::transmute::<*mut c_void, _>(b),
                        add_norm: std::mem::transmute::<*mut c_void, _>(c),
                        finish: std::mem::transmute::<*mut c_void, _>(d),
                        to_half: std::mem::transmute::<*mut c_void, _>(e),
                    }
                })
            }),
            silo_attend_split: nextsycl_core::silo(crate::ARCH).ok().flatten().and_then(|s| {
                let p = s.symbol("ns_mm3s_attend_split");
                // SAFETY: the symbol is declared in mm3s.h with SplitFn's signature.
                (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, SplitFn>(p) })
            }),
            silo_attend: nextsycl_core::silo(crate::ARCH).ok().flatten().and_then(|s| {
                let p = s.symbol("ns_mm3s_attend");
                // SAFETY: the symbol is declared in mm3s.h with AttendFn's signature.
                (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, AttendFn>(p) })
            }),
        })
    })
    .as_ref()
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextsycl_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextsycl_core::Error(format!("{what}: {why}")))
}

/// A copy of `n` bytes from `src` to `dst` (same GPU) on copy lane `lane`, after ticket `after`; its ticket
pub fn stream_copy(gpu: *mut c_void, dst: &nextsycl_core::DevBuf, src: &nextsycl_core::DevBuf, n: usize, after: &[i64], lane: i32) -> nextsycl_core::Result<i64> {
    let api = nextsycl_core::api()?;
    let deps: Vec<i64> = after.to_vec();
    let mut t = 0i64;
    // SAFETY: both buffers hold n bytes (callers view exactly the expert's bytes); t is an out-pointer to a local.
    let rc = unsafe { (api.stream_copy)(gpu, dst.ptr().cast(), src.ptr().cast_const().cast(), n, deps.as_ptr(), deps.len() as i32, lane, &mut t) };
    check(rc, "stream_copy")?;
    Ok(t)
}

/// A ticket for everything queued on the compute queue so far
pub fn mark(gpu: *mut c_void) -> nextsycl_core::Result<i64> {
    let api = nextsycl_core::api()?;
    let mut t = 0i64;
    // SAFETY: t is an out-pointer to a local.
    check(unsafe { (api.mark)(gpu, &mut t) }, "mark")?;
    Ok(t)
}

/// The compute queue waits, on the device, for ticket `t`
pub fn await_ticket(gpu: *mut c_void, t: i64) -> nextsycl_core::Result<()> {
    let api = nextsycl_core::api()?;
    // SAFETY: plain call.
    check(unsafe { (api.await_)(gpu, t) }, "await")
}
