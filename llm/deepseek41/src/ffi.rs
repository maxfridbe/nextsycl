//! The kernels this engine runs, bound by name: its own from its silo (`kernels/llm/deepseek41/silo/ds41.h`, built as
//! dist/silo/libnextsycl-deepseek41.so - the VQ products have no shared kernel, so the engine needs it) and, from the
//! language kind's kernel library, the generic ones of the glm5next engine's (`kernels/llm/glm5next/glm.h`: the mHC
//! mixes and Sinkhorn, norms, GEMMs, top-k, the q4_K decode products, the MoE combine).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type Gpu = *mut c_void;

pub struct Api {
    /// the shared ABI (ns.h): copies, tickets, timers
    pub shared: &'static nextsycl_core::Api,
    // glm.h
    pub dequant: unsafe extern "C" fn(Gpu, c_int, *const c_void, usize, *mut f32) -> c_int,
    pub gemm: unsafe extern "C" fn(Gpu, i64, i64, i64, *const f32, i64, *const f32, *mut f32, i64, c_int) -> c_int,
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
    pub topk: unsafe extern "C" fn(Gpu, *const f32, *mut i32, i64, i64, i64, i64) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(Gpu, c_int, *const c_void, i64, *mut u16) -> c_int,
    pub to_f16: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64) -> c_int,
    pub gather_f16: unsafe extern "C" fn(Gpu, *const f32, *const i32, *mut u16, i64, i64) -> c_int,
    pub gemm_f16: unsafe extern "C" fn(Gpu, i64, i64, i64, *const u16, i64, *const u16, *mut f32, i64, c_int) -> c_int,
    pub scatter_add: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const i32, *const f32, i64, i64) -> c_int,
    pub gather: unsafe extern "C" fn(Gpu, *const f32, *const i32, *mut f32, i64, i64) -> c_int,
    pub add: unsafe extern "C" fn(Gpu, *mut f32, *const f32, i64) -> c_int,
    pub moe_combine: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const i32, *const i32, *const f32, i64, i64) -> c_int,
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub quantize_q8_1: unsafe extern "C" fn(Gpu, *const f32, *mut c_void, i64, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(Gpu, c_int, *const c_void, *const c_void, *mut f32, i64, i64, i64) -> c_int,
    // ds41.h
    pub ds41_rope: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32, f32, f32, f32, c_int) -> c_int,
    pub ds41_pool: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const f32, *const f32, *mut f32, i64, i64, i64, i64, c_int, i64) -> c_int,
    pub ds41_ring_put: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64, i64, i64, i64) -> c_int,
    pub ds41_rows_f16: unsafe extern "C" fn(Gpu, *const f32, *mut u16, i64, i64, i64) -> c_int,
    #[allow(clippy::type_complexity)]
    pub ds41_attend: unsafe extern "C" fn(Gpu, *const f32, *const u16, i64, *const u16, i64, *const i32, i64, *const f32, *mut f32, i64, i64, i64, i64, f32) -> c_int,
    pub ds41_permute: unsafe extern "C" fn(Gpu, *const f32, *mut f32, i64, i64, i64) -> c_int,
    pub ds41_kcopy: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize, *mut i64) -> c_int,
    pub ds41_ticket_wait: unsafe extern "C" fn(Gpu, i64) -> c_int,
    pub ds41_quant_sim: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, i64, c_int) -> c_int,
    pub ds41_hc_collapse: unsafe extern "C" fn(Gpu, *const f32, *const f32, *mut f32, i64, i64) -> c_int,
    pub ds41_engram_rows: unsafe extern "C" fn(Gpu, *const u8, *mut f32, i64, i64) -> c_int,
    pub ds41_engram_gate: unsafe extern "C" fn(Gpu, *mut f32, *const f32, *const f32, i64, i64, f32) -> c_int,
    pub ds41_idx_score: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const u16, *mut f32, i64, i64, i64, i64, i64, i64, i64) -> c_int,
    pub ds41_block_max: unsafe extern "C" fn(Gpu, *const f32, *mut f32, i64, i64, i64, i64, i64, i64) -> c_int,
    pub ds41_idx_score_cand: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const u16, *const i32, i64, *mut f32, i64, i64, i64, i64, i64, i64) -> c_int,
    pub ds41_cand_pos: unsafe extern "C" fn(Gpu, *const i32, *const i32, *mut i32, i64, i64, i64, i64) -> c_int,
    pub ds41_route1: unsafe extern "C" fn(Gpu, *const f32, *const f32, i64, i64, f32, *const u64, u64, i64, i64, i64, *mut u8, *mut u8, *mut i32, *mut i32) -> c_int,
    pub ds41_scatter_u64: unsafe extern "C" fn(Gpu, *mut u64, *const i64, i64) -> c_int,
    pub ds41_att_post: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, i64, i64, i64, f32, f32, f32, f32, f32, f32) -> c_int,
    pub ds41_hc_pre_in: unsafe extern "C" fn(Gpu, *const f32, *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32, *mut f32,
                                             *mut f32, *mut f32, i64, i64, f32, f32, c_int) -> c_int,
    pub ds41_q4k_mv2: unsafe extern "C" fn(Gpu, *const c_void, i64, *const c_void, i64, i64, *const f32, *mut f32, *mut f32) -> c_int,
    pub ds41_gemv2: unsafe extern "C" fn(Gpu, *const f32, *const f32, i64, *const f32, i64, i64, *mut f32, *mut f32) -> c_int,
    pub gemm_batch: unsafe extern "C" fn(Gpu, i64, i64, i64, i64, *const f32, i64, i64, *const f32, i64, *mut f32, i64, i64, c_int) -> c_int,
    pub gemm_batch_nn: unsafe extern "C" fn(Gpu, i64, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, *mut f32, i64, i64, c_int) -> c_int,
    pub ds41_vit_rope: unsafe extern "C" fn(Gpu, *mut f32, *mut f32, i64, i64, i64, f32) -> c_int,
    pub ds41_softmax_rows: unsafe extern "C" fn(Gpu, *mut f32, i64, i64, f32) -> c_int,
    pub ds41_bias_act: unsafe extern "C" fn(Gpu, *mut f32, *const f32, i64, i64, c_int) -> c_int,
    pub ds41_silu_mul: unsafe extern "C" fn(Gpu, *const f32, *mut f32, i64, i64) -> c_int,
    pub ds41_kv_post: unsafe extern "C" fn(Gpu, *mut f32, *const f32, i64, i64, f32, i64, i64, f32, f32, f32, f32, f32, f32, *mut u16, i64) -> c_int,
    pub ds41_cand_mask: unsafe extern "C" fn(Gpu, *mut f32, *const i32, *mut u8, i64, i64, i64, i64, i64) -> c_int,
    pub ds41_vq_supported: unsafe extern "C" fn(i64, i64, i64) -> c_int,
    pub ds41_vq_dequant: unsafe extern "C" fn(Gpu, *const c_void, *const u8, i64, i64, i64, *mut c_void, c_int) -> c_int,
    pub ds41_vq_mmv: unsafe extern "C" fn(Gpu, *const u64, *const i32, i64, *const u8, *const f32, *const f32, *mut f32, i64, i64, i64, c_int) -> c_int,
    pub ds41_vq_codebook_f32: unsafe extern "C" fn(Gpu, *const u8, i64, *mut f32) -> c_int,
    pub ds41_vq_gemm: unsafe extern "C" fn(Gpu, *const c_void, *const c_void, i64, i64, i64, i64, *const c_void, *const c_void, i64, i64, *mut f32, i64) -> c_int,
    pub ds41_vq_lut: unsafe extern "C" fn(Gpu, *const u64, *const i32, *const i32, i64, i64, *const u8, *const f32, *mut f32, i64, i64, i64, i64, i64) -> c_int,
    pub ds41_fp8_dequant: unsafe extern "C" fn(Gpu, *const u8, i64, i64, i64, i64, *mut c_void, c_int) -> c_int,
    pub ds41_q4k_mv: unsafe extern "C" fn(Gpu, *const c_void, i64, i64, *const f32, i64, i64, i64, *mut f32, i64, i64) -> c_int,
    pub ds41_vq_codebook_f16: unsafe extern "C" fn(Gpu, *const u8, i64, *mut c_void) -> c_int,
    pub ds41_markov_step: unsafe extern "C" fn(Gpu, *const f32, *const f32, i64, *mut i32) -> c_int,
    pub ds41_gather_rows: unsafe extern "C" fn(Gpu, *const f32, *const i32, i64, i64, *mut f32) -> c_int,
    pub ds41_fp8_mmv: unsafe extern "C" fn(Gpu, *const u8, i64, i64, *const f32, i64, i64, i64, *mut f32, i64, i64) -> c_int,
}

// SAFETY: function pointers into the libraries, which stay loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// The table, bound once (an error when a library lacks a symbol: an older build)
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
        // this engine's own kernels: its silo's library (they resolve the shared part against the kind's, loaded first)
        let silo = nextsycl_core::silo(crate::ARCH)
            .map_err(|e| e.0)?
            .ok_or("deepseek41: no silo library (dist/silo/libnextsycl-deepseek41.so; build it: ./build.sh kernels)")?;
        macro_rules! own {
            ($name:literal) => {{
                let p = silo.symbol($name);
                if p.is_null() {
                    return Err(format!("{}: no {} (rebuild it: ./build.sh kernels)", silo.path.display(), $name));
                }
                // SAFETY: the symbol is declared in ds41.h with the field's signature.
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
            hc_mix: sym!("ns_hc_mix"),
            rms_norm: sym!("ns_rms_norm"),
            hc_pre: sym!("ns_hc_pre"),
            hc_pre_fused: sym!("ns_hc_pre_fused"),
            hc_post: sym!("ns_hc_post"),
            hc_mean: sym!("ns_hc_mean"),
            swiglu_clamp: sym!("ns_swiglu_clamp"),
            swiglu_gu_f16: sym!("ns_swiglu_gu_f16"),
            topk: sym!("ns_topk"),
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
            ds41_rope: own!("ns_ds41_rope"),
            ds41_pool: own!("ns_ds41_pool"),
            ds41_ring_put: own!("ns_ds41_ring_put"),
            ds41_rows_f16: own!("ns_ds41_rows_f16"),
            ds41_attend: own!("ns_ds41_attend"),
            ds41_permute: own!("ns_ds41_permute"),
            ds41_kcopy: own!("ns_ds41_kcopy"),
            ds41_ticket_wait: own!("ns_ds41_ticket_wait"),
            ds41_quant_sim: own!("ns_ds41_quant_sim"),
            ds41_hc_collapse: own!("ns_ds41_hc_collapse"),
            ds41_engram_rows: own!("ns_ds41_engram_rows"),
            ds41_engram_gate: own!("ns_ds41_engram_gate"),
            ds41_idx_score: own!("ns_ds41_idx_score"),
            ds41_block_max: own!("ns_ds41_block_max"),
            ds41_cand_mask: own!("ns_ds41_cand_mask"),
            ds41_route1: own!("ns_ds41_route1"),
            ds41_scatter_u64: own!("ns_ds41_scatter_u64"),
            ds41_kv_post: own!("ns_ds41_kv_post"),
            gemm_batch: sym!("ns_gemm_batch"),
            gemm_batch_nn: sym!("ns_gemm_batch_nn"),
            ds41_vit_rope: own!("ns_ds41_vit_rope"),
            ds41_softmax_rows: own!("ns_ds41_softmax_rows"),
            ds41_bias_act: own!("ns_ds41_bias_act"),
            ds41_silu_mul: own!("ns_ds41_silu_mul"),
            ds41_q4k_mv2: own!("ns_ds41_q4k_mv2"),
            ds41_gemv2: own!("ns_ds41_gemv2"),
            ds41_att_post: own!("ns_ds41_att_post"),
            ds41_hc_pre_in: own!("ns_ds41_hc_pre_in"),
            ds41_idx_score_cand: own!("ns_ds41_idx_score_cand"),
            ds41_cand_pos: own!("ns_ds41_cand_pos"),
            ds41_vq_supported: own!("ns_ds41_vq_supported"),
            ds41_vq_dequant: own!("ns_ds41_vq_dequant"),
            ds41_vq_mmv: own!("ns_ds41_vq_mmv"),
            ds41_vq_codebook_f32: own!("ns_ds41_vq_codebook_f32"),
            ds41_vq_lut: own!("ns_ds41_vq_lut"),
            ds41_vq_gemm: own!("ns_ds41_vq_gemm"),
            ds41_fp8_dequant: own!("ns_ds41_fp8_dequant"),
            ds41_fp8_mmv: own!("ns_ds41_fp8_mmv"),
            ds41_vq_codebook_f16: own!("ns_ds41_vq_codebook_f16"),
            ds41_markov_step: own!("ns_ds41_markov_step"),
            ds41_gather_rows: own!("ns_ds41_gather_rows"),
            ds41_q4k_mv: own!("ns_ds41_q4k_mv"),
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
