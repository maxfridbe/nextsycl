//! The kernels this engine runs, bound from the language kind's kernel library by name: its own
//! (`kernels/llm/qwen35/q35.h`) and the generic ones of the glm5next engine's (`kernels/llm/glm5next/glm.h`: the
//! products from the stored blocks, the norms, the conv + SiLU, the L2 norm, the delta-rule scan, SwiGLU).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type P = *const c_void;
type M = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    // glm.h
    pub dequant: unsafe extern "C" fn(G, c_int, P, usize, *mut f32) -> c_int,
    pub gemm: unsafe extern "C" fn(G, i64, i64, i64, *const f32, i64, *const f32, *mut f32, i64, c_int) -> c_int,
    pub rms_norm: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub conv_silu: unsafe extern "C" fn(G, *const f32, *mut f32, *const f32, *mut f32, i64, i64, c_int, *mut f32) -> c_int,
    pub l2_norm: unsafe extern "C" fn(G, *mut f32, i64, i64, f32) -> c_int,
    pub kda_scan: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, i64, i64, i64, *mut f32) -> c_int,
    pub swiglu_clamp: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, i64, f32) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(G, c_int, P, i64, *mut u16) -> c_int,
    pub to_f16: unsafe extern "C" fn(G, *const f32, *mut u16, i64) -> c_int,
    pub gemm_f16: unsafe extern "C" fn(G, i64, i64, i64, *const u16, i64, *const u16, *mut f32, i64, c_int) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub quantize_q8_1: unsafe extern "C" fn(G, *const f32, M, i64, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(G, c_int, P, P, *mut f32, i64, i64, i64) -> c_int,
    // q35.h
    pub qk_norm_rope: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32, i64, i64, i64, i64, *const f32, f32, i64, f32, i64) -> c_int,
    pub kv_store: unsafe extern "C" fn(G, *const f32, i64, *const f32, i64, i64, i64, i64, i64, i64, M, M) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub attn: unsafe extern "C" fn(G, *const f32, i64, P, P, i64, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    pub gate_mul: unsafe extern "C" fn(G, *mut f32, *const f32, i64, i64, i64, i64) -> c_int,
    pub gdn_gates: unsafe extern "C" fn(G, *const f32, *mut f32, *const f32, *const f32, *mut f32, i64, i64, i64) -> c_int,
    pub expand: unsafe extern "C" fn(G, *const f32, i64, *mut f32, i64, i64, i64, i64) -> c_int,
    pub gdn_out: unsafe extern "C" fn(G, *const f32, *const f32, i64, *const f32, *mut f32, i64, i64, i64, f32) -> c_int,
    /// the greedy pick on the GPU (not used yet: the logits are read back for the sampler)
    #[allow(dead_code)]
    pub argmax: unsafe extern "C" fn(G, *const f32, i64, i64, *mut i32) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (rebuild it: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in glm.h / q35.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            dequant: sym!("ns_dequant"),
            gemm: sym!("ns_gemm"),
            rms_norm: sym!("ns_rms_norm"),
            conv_silu: sym!("ns_conv_silu"),
            l2_norm: sym!("ns_l2_norm"),
            kda_scan: sym!("ns_kda_scan"),
            swiglu_clamp: sym!("ns_swiglu_clamp"),
            dequant_f16: sym!("ns_dequant_f16"),
            to_f16: sym!("ns_to_f16"),
            gemm_f16: sym!("ns_gemm_f16"),
            add: sym!("ns_add"),
            q8_1_bytes: sym!("ns_q8_1_bytes"),
            mmvq_supported: sym!("ns_mmvq_supported"),
            quantize_q8_1: sym!("ns_quantize_q8_1"),
            mmvq: sym!("ns_mmvq"),
            qk_norm_rope: sym!("ns_q35_qk_norm_rope"),
            kv_store: sym!("ns_q35_kv_store"),
            attn_scratch: sym!("ns_q35_attn_scratch"),
            attn: sym!("ns_q35_attn"),
            gate_mul: sym!("ns_q35_gate_mul"),
            gdn_gates: sym!("ns_q35_gdn_gates"),
            expand: sym!("ns_q35_expand"),
            gdn_out: sym!("ns_q35_gdn_out"),
            argmax: sym!("ns_q35_argmax"),
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

/// The silo's kernels (kernels/llm/qwen35/silo/silo.h, dist/silo/libnextsycl-qwen35.so), when it is there
pub struct Silo {
    pub path: std::path::PathBuf,
    pub mmvq_supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(G, c_int, P, P, *mut f32, i64, i64, i64) -> c_int,
    pub attn_prompt_supported: unsafe extern "C" fn(i64, i64, i64, i64) -> c_int,
    pub attn_prompt: unsafe extern "C" fn(G, *const f32, i64, P, P, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub attn_decode_supported: unsafe extern "C" fn(i64, i64, i64, i64) -> c_int,
    pub attn_decode_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub attn_decode: unsafe extern "C" fn(G, *const f32, i64, P, P, i64, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    pub dequant_f16_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(G, c_int, P, i64, *mut u16) -> c_int,
}

/// The silo (None: the shared kernels for everything)
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static SILO: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    SILO.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(crate::ARCH).map_err(|e| e.0)? else { return Ok(None) };
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
            mmvq_supported: sym!("ns_q35_mmvq_supported"),
            mmvq: sym!("ns_q35_mmvq"),
            attn_prompt_supported: sym!("ns_q35_attn_prompt_supported"),
            attn_prompt: sym!("ns_q35_attn_prompt"),
            attn_decode_supported: sym!("ns_q35_attn_decode_supported"),
            attn_decode_scratch: sym!("ns_q35_attn_decode_scratch"),
            attn_decode: sym!("ns_q35_attn_decode"),
            dequant_f16_supported: sym!("ns_q35_dequant_f16_supported"),
            dequant_f16: sym!("ns_q35_dequant_f16"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| nextsycl_core::Error(e.clone()))
}
