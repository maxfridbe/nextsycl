//! This engine's own C ABI (`kernels/llm/glm5next/glm.h`), bound from the kernel library by name.

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
    pub layer_norm: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub hc_pre: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32, *mut f32, i64, i64, f32, c_int) -> c_int,
    #[allow(clippy::type_complexity)]
    pub hc_pre_fused: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32, *mut f32,
                                           *mut f32, *mut f32, i64, i64, f32, f32, c_int) -> c_int,
    pub hc_post: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *mut f32, i64, i64) -> c_int,
    pub hc_mean: unsafe extern "C" fn(Gpu, *const f32, *mut f32, i64, i64) -> c_int,
    pub conv_silu: unsafe extern "C" fn(Gpu, *const f32, *mut f32, *const f32, *mut f32, i64, i64, c_int, *mut f32) -> c_int,
    pub l2_norm: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, f32) -> c_int,
    pub kda_gate: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const f32, i64, i64, i64, f32) -> c_int,
    pub sigmoid: unsafe extern "C" fn(Gpu, *mut f32, i64) -> c_int,
    pub exp: unsafe extern "C" fn(Gpu, *mut f32, i64) -> c_int,
    pub kda_scan: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, i64, i64, i64, *mut f32) -> c_int,
    pub kda_out: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *mut f32, i64, i64, i64, f32) -> c_int,
    pub swiglu_clamp: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, f32) -> c_int,
    pub swiglu_gu_f16: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64, i64, f32) -> c_int,
    pub mla_attend: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, i64, i64, i64, f32) -> c_int,
    pub idx_pool: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const f32, *const f32, *mut f32, i64, i64, i64) -> c_int,
    pub idx_score: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, i64, i64, i64, i64, i64) -> c_int,
    pub topk: unsafe extern "C" fn(Gpu, *const f32, *mut i32, i64, i64, i64, i64) -> c_int,
    pub mla_cells: unsafe extern "C" fn(Gpu, *const i32, *const i32, i64, i64, i64, *mut i32, *mut i32, i64) -> c_int,
    pub softmax_masked: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, i64, *const i32, f32) -> c_int,
    pub gemm_batch_nn: unsafe extern "C" fn(Gpu, i64, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, *mut f32, i64, i64, c_int) -> c_int,
    #[allow(clippy::type_complexity)]
    pub moe_fused_gu: unsafe extern "C" fn(Gpu, *const u16, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    pub moe_fused_down: unsafe extern "C" fn(Gpu, *const u16, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    pub moe_esimd_gu_max: unsafe extern "C" fn(Gpu) -> c_int,
    pub moe_fused_gu_esimd: unsafe extern "C" fn(Gpu, *const u16, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    pub gemm_batch_h: unsafe extern "C" fn(Gpu, i64, c_int, i64, i64, i64, *const u16, i64, i64, *const u16, i64, i64, *mut f32, i64, i64, c_int) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(Gpu, c_int, *const c_void, i64, *mut u16) -> c_int,
    pub to_f16: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64) -> c_int,
    pub gather_f16: unsafe extern "C" fn(Gpu, *const f32, *const i32, *mut u16, i64, i64) -> c_int,
    pub gather_h: unsafe extern "C" fn(Gpu, *const u16, *const i32, *mut u16, i64, i64) -> c_int,
    pub to_q8row: unsafe extern "C" fn(Gpu, *const f32, *mut c_void, i64, i64) -> c_int,
    pub gather_q8_h: unsafe extern "C" fn(Gpu, *const c_void, *const i32, *mut u16, i64, i64) -> c_int,
    pub mla_attend_sel_q8: unsafe extern "C" fn(Gpu, *const f32, *const c_void, *mut f32, i64, i64, i64, i64, f32, *const i32, *const i32, i64) -> c_int,
    pub gemm_f16: unsafe extern "C" fn(Gpu, i64, i64, i64, *const u16, i64, *const u16, *mut f32, i64, c_int) -> c_int,
    pub mla_attend_sel: unsafe extern "C" fn(Gpu, *const f32, *const u16, *mut f32, i64, i64, i64, i64, f32, *const i32, *const i32, i64) -> c_int,
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
                // SAFETY: the symbol is declared in glm.h with the field's signature.
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
            layer_norm: sym!("ns_layer_norm"),
            hc_pre: sym!("ns_hc_pre"),
            hc_pre_fused: sym!("ns_hc_pre_fused"),
            hc_post: sym!("ns_hc_post"),
            hc_mean: sym!("ns_hc_mean"),
            conv_silu: sym!("ns_conv_silu"),
            l2_norm: sym!("ns_l2_norm"),
            kda_gate: sym!("ns_kda_gate"),
            sigmoid: sym!("ns_sigmoid"),
            exp: sym!("ns_exp"),
            kda_scan: sym!("ns_kda_scan"),
            kda_out: sym!("ns_kda_out"),
            swiglu_clamp: sym!("ns_swiglu_clamp"),
            swiglu_gu_f16: sym!("ns_swiglu_gu_f16"),
            mla_attend: sym!("ns_mla_attend"),
            idx_pool: sym!("ns_idx_pool"),
            idx_score: sym!("ns_idx_score"),
            topk: sym!("ns_topk"),
            mla_cells: sym!("ns_mla_cells"),
            softmax_masked: sym!("ns_softmax_masked"),
            gemm_batch_nn: sym!("ns_gemm_batch_nn"),
            moe_fused_gu: sym!("ns_moe_fused_gu"),
            moe_fused_down: sym!("ns_moe_fused_down"),
            moe_esimd_gu_max: sym!("ns_moe_esimd_gu_max"),
            moe_fused_gu_esimd: sym!("ns_moe_fused_gu_esimd"),
            gemm_batch_h: sym!("ns_gemm_batch_h"),
            dequant_f16: sym!("ns_dequant_f16"),
            to_f16: sym!("ns_to_f16"),
            gather_f16: sym!("ns_gather_f16"),
            gather_h: sym!("ns_gather_h"),
            to_q8row: sym!("ns_to_q8row"),
            gather_q8_h: sym!("ns_gather_q8_h"),
            mla_attend_sel_q8: sym!("ns_mla_attend_sel_q8"),
            gemm_f16: sym!("ns_gemm_f16"),
            mla_attend_sel: sym!("ns_mla_attend_sel"),
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
