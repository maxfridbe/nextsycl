//! The kernels this engine runs, bound from the language kind's kernel library by name: its own
//! (`kernels/llm/nemotronh/nh.h`: the Mamba-2 conv, scan and gated norm, relu^2, the attention) and the generic ones
//! of the glm5next engine's (`kernels/llm/glm5next/glm.h`: the products from the stored blocks, the RMS norm, the
//! MoE combine).

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
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub quantize_q8_1: unsafe extern "C" fn(G, *const f32, M, i64, i64) -> c_int,
    pub mmvq: unsafe extern "C" fn(G, c_int, P, P, *mut f32, i64, i64, i64) -> c_int,
    pub moe_combine: unsafe extern "C" fn(G, *mut f32, *const f32, *const i32, *const i32, *const f32, i64, i64) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(G, c_int, P, i64, *mut u16) -> c_int,
    pub to_f16: unsafe extern "C" fn(G, *const f32, *mut u16, i64) -> c_int,
    pub gemm_f16: unsafe extern "C" fn(G, i64, i64, i64, *const u16, i64, *const u16, *mut f32, i64, c_int) -> c_int,
    pub gather_h: unsafe extern "C" fn(G, *const u16, *const i32, *mut u16, i64, i64) -> c_int,
    // nh.h
    pub conv: unsafe extern "C" fn(G, *const f32, i64, *mut f32, *const f32, *const f32, *mut f32, i64, i64, c_int, *mut f32) -> c_int,
    pub ssm: unsafe extern "C" fn(G, *const f32, i64, *const f32, i64, *const f32, *const f32, *const f32, *mut f32, *mut f32, i64, i64, i64, i64, i64, *mut f32)
                                 -> c_int,
    pub layer_norm: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    /// the greedy pick on the GPU (the qwen35 engine's kernel: the MTP drafts)
    pub argmax: unsafe extern "C" fn(G, *const f32, i64, i64, *mut i32) -> c_int,
    pub gated_norm: unsafe extern "C" fn(G, *const f32, *const f32, i64, *const f32, *mut f32, i64, i64, i64, f32) -> c_int,
    pub relu2: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
    pub kv_store: unsafe extern "C" fn(G, *const f32, i64, *const f32, i64, i64, i64, i64, i64, i64, M, M) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub attn: unsafe extern "C" fn(G, *const f32, i64, P, P, i64, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    pub q6k_f16: unsafe extern "C" fn(G, P, i64, *mut u16) -> c_int,
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
                // SAFETY: the symbol is declared in glm.h / nh.h with the field's signature.
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
            add: sym!("ns_add"),
            q8_1_bytes: sym!("ns_q8_1_bytes"),
            mmvq_supported: sym!("ns_mmvq_supported"),
            quantize_q8_1: sym!("ns_quantize_q8_1"),
            mmvq: sym!("ns_mmvq"),
            moe_combine: sym!("ns_moe_combine"),
            dequant_f16: sym!("ns_dequant_f16"),
            to_f16: sym!("ns_to_f16"),
            gemm_f16: sym!("ns_gemm_f16"),
            gather_h: sym!("ns_gather_h"),
            conv: sym!("ns_nh_conv"),
            layer_norm: sym!("ns_layer_norm"),
            argmax: sym!("ns_q35_argmax"),
            ssm: sym!("ns_nh_ssm"),
            gated_norm: sym!("ns_nh_gated_norm"),
            relu2: sym!("ns_nh_relu2"),
            kv_store: sym!("ns_nh_kv_store"),
            attn_scratch: sym!("ns_nh_attn_scratch"),
            attn: sym!("ns_nh_attn"),
            q6k_f16: sym!("ns_nh_q6k_f16"),
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

/// The silo's kernels (kernels/llm/nemotronh/silo/nhs.h, dist/silo/libnextsycl-nemotronh.so), when it is there
pub struct Silo {
    pub path: std::path::PathBuf,
    pub route: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, c_int, f32, *mut i32, *mut f32, *mut i32) -> c_int,
    pub q8_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub quantize_q8: unsafe extern "C" fn(G, *const f32, M, i64, i64) -> c_int,
    pub multi_supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    pub multi: unsafe extern "C" fn(G, c_int, P, i64, *const i32, *const i32, i64, P, i64, *mut f32, i64, i64) -> c_int,
    pub packed: unsafe extern "C" fn(c_int, i64) -> c_int,
    pub packed_row_bytes: unsafe extern "C" fn(c_int, i64) -> i64,
    pub pack: unsafe extern "C" fn(G, c_int, P, M, i64, i64) -> c_int,
    pub packed_f16: unsafe extern "C" fn(G, c_int, P, *mut u16, i64, i64) -> c_int,
    pub wide_f16: unsafe extern "C" fn(G, c_int, P, *mut u16, i64, i64) -> c_int,
    pub gather: unsafe extern "C" fn(G, P, i64, *const i32, i64, M) -> c_int,
    pub attn_prompt_supported: unsafe extern "C" fn(i64, i64, i64, i64) -> c_int,
    pub attn_prompt: unsafe extern "C" fn(G, *const f32, i64, P, P, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub attn_decode_supported: unsafe extern "C" fn(i64, i64, i64, i64) -> c_int,
    pub attn_decode_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub attn_decode: unsafe extern "C" fn(G, *const f32, i64, P, P, i64, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
}

// SAFETY: function pointers into the silo library, which stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo (None: the shared kernels and the host's routing)
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static SILO: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    SILO.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo("nemotronh").map_err(|e| e.0)? else { return Ok(None) };
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{}: no {} (rebuild it: ./build.sh kernels)", lib.path.display(), $name));
                }
                // SAFETY: the symbol is declared in nhs.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Some(Silo {
            path: lib.path.clone(),
            route: sym!("ns_nhs_route"),
            q8_bytes: sym!("ns_nhs_q8_bytes"),
            quantize_q8: sym!("ns_nhs_quantize_q8"),
            multi_supported: sym!("ns_nhs_multi_supported"),
            multi: sym!("ns_nhs_multi"),
            packed: sym!("ns_nhs_packed"),
            packed_row_bytes: sym!("ns_nhs_packed_row_bytes"),
            pack: sym!("ns_nhs_pack"),
            packed_f16: sym!("ns_nhs_packed_f16"),
            wide_f16: sym!("ns_nhs_wide_f16"),
            gather: sym!("ns_nhs_gather"),
            attn_prompt_supported: sym!("ns_nhs_attn_prompt_supported"),
            attn_prompt: sym!("ns_nhs_attn_prompt"),
            attn_decode_supported: sym!("ns_nhs_attn_decode_supported"),
            attn_decode_scratch: sym!("ns_nhs_attn_decode_scratch"),
            attn_decode: sym!("ns_nhs_attn_decode"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| nextsycl_core::Error(e.clone()))
}
