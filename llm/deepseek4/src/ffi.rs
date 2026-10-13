//! The kernels this engine runs, bound by name: its own from its silo (`kernels/llm/deepseek4/silo/ds4.h`, built as
//! dist/silo/libnextsycl-deepseek4.so - there is no shared fallback for them, so the engine needs it) and, from the
//! language kind's kernel library, the generic ones of the glm5next engine's (`kernels/llm/glm5next/glm.h`: the
//! mHC halves, norms, GEMMs, the indexer's scores and top-k, the decode products, the grouped experts).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type Gpu = *mut c_void;

pub struct Api {
    /// the shared ABI (ns.h), for the calls both engines make
    pub shared: &'static nextsycl_core::Api,
    // the bring-up kernels (glm.cpp)
    pub dequant: unsafe extern "C" fn(Gpu, c_int, *const c_void, usize, *mut f32) -> c_int,
    pub gemm: unsafe extern "C" fn(Gpu, i64, i64, i64, *const f32, i64, *const f32, *mut f32, i64, c_int) -> c_int,
    pub gemm_batch: unsafe extern "C" fn(Gpu, i64, i64, i64, i64, *const f32, i64, i64, *const f32, i64, *mut f32, i64, i64, c_int) -> c_int,
    pub hc_mix: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, *mut f32, i64, i64, f32) -> c_int,
    pub rms_norm: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub hc_pre: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32, *mut f32, i64, i64, f32, c_int) -> c_int,
    #[allow(clippy::type_complexity)]
    pub hc_pre_fused: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32, *mut f32,
                                           *mut f32, *mut f32, i64, i64, f32, f32, c_int) -> c_int,
    pub hc_post: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *mut f32, i64, i64) -> c_int,
    pub hc_mean: unsafe extern "C" fn(Gpu, *const f32, *mut f32, i64, i64) -> c_int,
    pub swiglu_clamp: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, f32) -> c_int,
    pub swiglu_gu_f16: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64, i64, f32) -> c_int,
    pub idx_score: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, i64, i64, i64, i64, i64) -> c_int,
    pub topk: unsafe extern "C" fn(Gpu, *const f32, *mut i32, i64, i64, i64, i64) -> c_int,
    #[allow(clippy::type_complexity)]
    pub moe_fused_gu: unsafe extern "C" fn(Gpu, *const u16, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    pub moe_fused_down: unsafe extern "C" fn(Gpu, *const u16, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    pub moe_esimd_gu_max: unsafe extern "C" fn(Gpu) -> c_int,
    pub moe_fused_gu_esimd: unsafe extern "C" fn(Gpu, *const u16, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    pub gemm_batch_h: unsafe extern "C" fn(Gpu, i64, c_int, i64, i64, i64, *const u16, i64, i64, *const u16, i64, i64, *mut f32, i64, i64, c_int) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(Gpu, c_int, *const c_void, i64, *mut u16) -> c_int,
    pub to_f16: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64) -> c_int,
    pub gather_f16: unsafe extern "C" fn(Gpu, *const f32, *const i32, *mut u16, i64, i64) -> c_int,
    pub gemm_f16: unsafe extern "C" fn(Gpu, i64, i64, i64, *const u16, i64, *const u16, *mut f32, i64, c_int) -> c_int,
    pub scatter_add: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const i32, *const f32, i64, i64) -> c_int,
    pub gather: unsafe extern "C" fn(Gpu, *const f32, *const i32, *mut f32, i64, i64) -> c_int,
    pub add: unsafe extern "C" fn(Gpu, *mut f32, *const f32, i64) -> c_int,
    pub moe_combine: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const i32, *const i32, *const f32, i64, i64) -> c_int,
    // decode-width products (mmvq.cpp)
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub quantize_q8_1: unsafe extern "C" fn(Gpu, *const f32, *mut c_void, i64, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(Gpu, c_int, *const c_void, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    pub moe_grouped_supported: unsafe extern "C" fn(c_int, c_int, i64, i64) -> c_int,
    pub moe_scratch_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub moe_grouped: unsafe extern "C" fn(Gpu, c_int, c_int, i64, i64, *const u64, *const i32, *const i32, *const i32, *const i32, i64, i64,
                                          *const c_void, *mut c_void, *mut f32, f32, c_int, c_int) -> c_int,
    // ds4.h
    pub ds4_rope: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32, f32, f32, f32, c_int) -> c_int,
    pub ds4_pool: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const f32, *const f32, *mut f32, i64, i64, i64, i64, c_int, i64) -> c_int,
    pub ds4_ring_put: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64, i64, i64, i64) -> c_int,
    pub ds4_rows_f16: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64, i64, i64) -> c_int,
    #[allow(clippy::type_complexity)]
    pub ds4_attend: unsafe extern "C" fn(Gpu, *const f32, *const u16, i64, *const u16, i64, *const i32, i64, *const f32, *mut f32, i64, i64, i64, i64, f32) -> c_int,
    pub ds4_hc_head: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub ds4_permute: unsafe extern "C" fn(Gpu, *const f32, *mut f32, i64, i64, i64) -> c_int,
    pub ds4_mxfp4_dequant: unsafe extern "C" fn(Gpu, *const c_void, i64, *mut f32) -> c_int,
    pub ds4_mxfp4_dequant_f16: unsafe extern "C" fn(Gpu, *const c_void, i64, *mut u16) -> c_int,
    pub ds4_mxfp4_mmvq: unsafe extern "C" fn(Gpu, *const c_void, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    pub ds4_q8_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub ds4_quantize_q8: unsafe extern "C" fn(Gpu, *const f32, *mut c_void, i64, i64) -> c_int,
    pub ds4_multi_supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    pub ds4_mmvq_multi: unsafe extern "C" fn(Gpu, c_int, *const u64, *const i32, i64, *const c_void, i64, *mut f32, i64, i64) -> c_int,
    pub ds4_mmvq_multi_var: unsafe extern "C" fn(Gpu, c_int, *const u64, *const i32, *const i32, i64, *const c_void, i64, *mut f32, i64, i64) -> c_int,
    pub ds4_kcopy: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize, *mut i64) -> c_int,
    pub ds4_ticket_wait: unsafe extern "C" fn(Gpu, i64) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// The table, bound once (an error when the library lacks a symbol: an older build)
pub fn api() -> nextsycl_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let shared = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = shared.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (rebuild it: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in glm.h / ds4.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        // this engine's own kernels: its silo's library (they resolve the shared part against the kind's, loaded first)
        let silo = nextsycl_core::silo(crate::ARCH)
            .map_err(|e| e.0)?
            .ok_or("deepseek4: no silo library (dist/silo/libnextsycl-deepseek4.so; build it: ./build.sh kernels)")?;
        macro_rules! own {
            ($name:literal) => {{
                let p = silo.symbol($name);
                if p.is_null() {
                    return Err(format!("{}: no {} (rebuild it: ./build.sh kernels)", silo.path.display(), $name));
                }
                // SAFETY: the symbol is declared in ds4.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            shared,
            dequant: sym!("ns_dequant"),
            gemm: sym!("ns_gemm"),
            gemm_batch: sym!("ns_gemm_batch"),
            hc_mix: sym!("ns_hc_mix"),
            rms_norm: sym!("ns_rms_norm"),
            hc_pre: sym!("ns_hc_pre"),
            hc_pre_fused: sym!("ns_hc_pre_fused"),
            hc_post: sym!("ns_hc_post"),
            hc_mean: sym!("ns_hc_mean"),
            swiglu_clamp: sym!("ns_swiglu_clamp"),
            swiglu_gu_f16: sym!("ns_swiglu_gu_f16"),
            idx_score: sym!("ns_idx_score"),
            topk: sym!("ns_topk"),
            moe_fused_gu: sym!("ns_moe_fused_gu"),
            moe_fused_down: sym!("ns_moe_fused_down"),
            moe_esimd_gu_max: sym!("ns_moe_esimd_gu_max"),
            moe_fused_gu_esimd: sym!("ns_moe_fused_gu_esimd"),
            gemm_batch_h: sym!("ns_gemm_batch_h"),
            dequant_f16: sym!("ns_dequant_f16"),
            to_f16: sym!("ns_to_f16"),
            gather_f16: sym!("ns_gather_f16"),
            gemm_f16: sym!("ns_gemm_f16"),
            scatter_add: sym!("ns_scatter_add"),
            gather: sym!("ns_gather"),
            add: sym!("ns_add"),
            moe_combine: sym!("ns_moe_combine"),
            q8_1_bytes: sym!("ns_q8_1_bytes"),
            mmvq_supported: sym!("ns_mmvq_supported"),
            quantize_q8_1: sym!("ns_quantize_q8_1"),
            mmvq: sym!("ns_mmvq"),
            moe_grouped_supported: sym!("ns_moe_grouped_supported"),
            moe_scratch_bytes: sym!("ns_moe_scratch_bytes"),
            moe_grouped: sym!("ns_moe_grouped"),
            ds4_rope: own!("ns_ds4_rope"),
            ds4_pool: own!("ns_ds4_pool"),
            ds4_ring_put: own!("ns_ds4_ring_put"),
            ds4_rows_f16: own!("ns_ds4_rows_f16"),
            ds4_attend: own!("ns_ds4_attend"),
            ds4_hc_head: own!("ns_ds4_hc_head"),
            ds4_permute: own!("ns_ds4_permute"),
            ds4_mxfp4_dequant: own!("ns_ds4_mxfp4_dequant"),
            ds4_mxfp4_dequant_f16: own!("ns_ds4_mxfp4_dequant_f16"),
            ds4_mxfp4_mmvq: own!("ns_ds4_mxfp4_mmvq"),
            ds4_q8_bytes: own!("ns_ds4_q8_bytes"),
            ds4_quantize_q8: own!("ns_ds4_quantize_q8"),
            ds4_multi_supported: own!("ns_ds4_multi_supported"),
            ds4_mmvq_multi: own!("ns_ds4_mmvq_multi"),
            ds4_mmvq_multi_var: own!("ns_ds4_mmvq_multi_var"),
            ds4_kcopy: own!("ns_ds4_kcopy"),
            ds4_ticket_wait: own!("ns_ds4_ticket_wait"),
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
