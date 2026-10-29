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
    pub scatter_add: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const i32, *const f32, i64, i64) -> c_int,
    pub add: unsafe extern "C" fn(Gpu, *mut f32, *const f32, i64) -> c_int,
    pub gather: unsafe extern "C" fn(Gpu, *const f32, *const i32, *mut f32, i64, i64) -> c_int,
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub quantize_q8_1: unsafe extern "C" fn(Gpu, *const f32, *mut c_void, i64, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(Gpu, c_int, *const c_void, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    // mm3.h
    pub rope: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, i64, i64, i64, i64, f32) -> c_int,
    pub swiglu_oai: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, f32, f32) -> c_int,
    pub idx_scores: unsafe extern "C" fn(Gpu, *const f32, *const u16, *mut f32, i64, i64, i64, i64, i64) -> c_int,
    pub idx_topk: unsafe extern "C" fn(Gpu, *mut f32, *mut i32, i64, i64, i64, i64) -> c_int,
    #[allow(clippy::type_complexity)]
    pub attend: unsafe extern "C" fn(Gpu, *const f32, *const u16, *const u16, *const i32, i64, *mut f32, i64, i64, i64, i64, f32) -> c_int,
}

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
            scatter_add: sym!("ns_scatter_add"),
            add: sym!("ns_add"),
            gather: sym!("ns_gather"),
            q8_1_bytes: sym!("ns_q8_1_bytes"),
            mmvq_supported: sym!("ns_mmvq_supported"),
            quantize_q8_1: sym!("ns_quantize_q8_1"),
            mmvq: sym!("ns_mmvq"),
            rope: sym!("ns_llm_minimaxm3_rope"),
            swiglu_oai: sym!("ns_llm_minimaxm3_swiglu_oai"),
            idx_scores: sym!("ns_llm_minimaxm3_idx_scores"),
            idx_topk: sym!("ns_llm_minimaxm3_idx_topk"),
            attend: sym!("ns_llm_minimaxm3_attend"),
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
