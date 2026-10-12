//! The kernels this engine runs, bound from the language kind's kernel library by name: its own
//! (`kernels/llm/deepseek2/ds2.h`) and the generic ones of the glm5next engine's (`kernels/llm/glm5next/glm.h`: the
//! dense types' decode products, the norms, the GEMMs, the masked softmax, gather / scatter / combine).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type P = *const c_void;
type M = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    /// the shared ABI (ns.h): copies, tickets, timers
    pub shared: &'static nextsycl_core::Api,
    // glm.h
    pub dequant: unsafe extern "C" fn(G, c_int, P, usize, *mut f32) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(G, c_int, P, i64, *mut u16) -> c_int,
    pub gemm: unsafe extern "C" fn(G, i64, i64, i64, *const f32, i64, *const f32, *mut f32, i64, c_int) -> c_int,
    pub gemm_f16: unsafe extern "C" fn(G, i64, i64, i64, *const u16, i64, *const u16, *mut f32, i64, c_int) -> c_int,
    pub gemm_batch_h: unsafe extern "C" fn(G, i64, c_int, i64, i64, i64, *const u16, i64, i64, *const u16, i64, i64, *mut f32, i64, i64, c_int) -> c_int,
    pub rms_norm: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub swiglu_clamp: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, i64, f32) -> c_int,
    pub swiglu_gu_f16: unsafe extern "C" fn(G, *const f32, *mut u16, i64, i64, f32) -> c_int,
    pub softmax_masked: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, *const i32, f32) -> c_int,
    pub to_f16: unsafe extern "C" fn(G, *const f32, *mut u16, i64) -> c_int,
    pub gather_f16: unsafe extern "C" fn(G, *const f32, *const i32, *mut u16, i64, i64) -> c_int,
    pub scatter_add: unsafe extern "C" fn(G, *mut f32, *const f32, *const i32, *const f32, i64, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub moe_combine: unsafe extern "C" fn(G, *mut f32, *const f32, *const i32, *const i32, *const f32, i64, i64) -> c_int,
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub quantize_q8_1: unsafe extern "C" fn(G, *const f32, M, i64, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(G, c_int, P, P, *mut f32, i64, i64, i64) -> c_int,
    // ds2.h
    pub rope: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32, f32, f32, f32, *mut f32, i64, i64) -> c_int,
    pub kv_store: unsafe extern "C" fn(G, *const f32, *const f32, f32, *mut u16, i64, i64, i64, i64, f32, f32, f32, f32, f32, f32) -> c_int,
    pub mla_decode: unsafe extern "C" fn(G, *const f32, *const u16, *const u16, *const u16, *mut f32, i64, i64, i64, f32) -> c_int,
    pub mla_decode_q6k: unsafe extern "C" fn(G, *const f32, P, *const u16, *mut f32, i64, i64, i64, f32) -> c_int,
    pub swiglu_rows: unsafe extern "C" fn(G, *const f32, *mut f32, i64, i64) -> c_int,
    pub own_dequant_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub own_dequant: unsafe extern "C" fn(G, c_int, P, i64, *mut f32) -> c_int,
    pub own_dequant_f16: unsafe extern "C" fn(G, c_int, P, i64, *mut u16) -> c_int,
    pub kcopy: unsafe extern "C" fn(G, M, P, usize, i64, *mut i64) -> c_int,
    pub fill: unsafe extern "C" fn(G, *mut f32, f32, i64) -> c_int,
    pub ticket_wait: unsafe extern "C" fn(G, i64) -> c_int,
    pub ticket_done: unsafe extern "C" fn(G, i64) -> c_int,
    pub q8_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub quantize_q8: unsafe extern "C" fn(G, *const f32, M, i64, i64) -> c_int,
    pub own_mmvq_supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    pub own_mmvq: unsafe extern "C" fn(G, c_int, P, P, *mut f32, i64, i64, i64) -> c_int,
    pub mmvq_multi: unsafe extern "C" fn(G, c_int, *const u64, *const i32, i64, P, i64, *mut f32, i64, i64) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

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
                // SAFETY: the symbol is declared in glm.h / ds2.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            shared,
            dequant: sym!("ns_dequant"),
            dequant_f16: sym!("ns_dequant_f16"),
            gemm: sym!("ns_gemm"),
            gemm_f16: sym!("ns_gemm_f16"),
            gemm_batch_h: sym!("ns_gemm_batch_h"),
            rms_norm: sym!("ns_rms_norm"),
            swiglu_clamp: sym!("ns_swiglu_clamp"),
            swiglu_gu_f16: sym!("ns_swiglu_gu_f16"),
            softmax_masked: sym!("ns_softmax_masked"),
            to_f16: sym!("ns_to_f16"),
            gather_f16: sym!("ns_gather_f16"),
            scatter_add: sym!("ns_scatter_add"),
            add: sym!("ns_add"),
            moe_combine: sym!("ns_moe_combine"),
            q8_1_bytes: sym!("ns_q8_1_bytes"),
            mmvq_supported: sym!("ns_mmvq_supported"),
            quantize_q8_1: sym!("ns_quantize_q8_1"),
            mmvq: sym!("ns_mmvq"),
            rope: sym!("ns_ds2_rope"),
            kv_store: sym!("ns_ds2_kv_store"),
            mla_decode: sym!("ns_ds2_mla_decode"),
            mla_decode_q6k: sym!("ns_ds2_mla_decode_q6k"),
            swiglu_rows: sym!("ns_ds2_swiglu_rows"),
            own_dequant_supported: sym!("ns_ds2_dequant_supported"),
            own_dequant: sym!("ns_ds2_dequant"),
            own_dequant_f16: sym!("ns_ds2_dequant_f16"),
            kcopy: sym!("ns_ds2_kcopy"),
            fill: sym!("ns_ds2_fill"),
            ticket_wait: sym!("ns_ds2_ticket_wait"),
            ticket_done: sym!("ns_ds2_ticket_done"),
            q8_bytes: sym!("ns_ds2_q8_bytes"),
            quantize_q8: sym!("ns_ds2_quantize_q8"),
            own_mmvq_supported: sym!("ns_ds2_mmvq_supported"),
            own_mmvq: sym!("ns_ds2_mmvq"),
            mmvq_multi: sym!("ns_ds2_mmvq_multi"),
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
