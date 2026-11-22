//! The kernels this engine runs, bound from the language kind's kernel library by name: its own
//! (`kernels/llm/inkling2/ink.h`) and the glm5next engine's generic ones (`kernels/llm/glm5next/glm.h`: the RMS norm,
//! the float32 and half GEMMs, float32 -> half).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type P = *const c_void;
type M = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    // glm.h
    pub gemm: unsafe extern "C" fn(G, i64, i64, i64, *const f32, i64, *const f32, *mut f32, i64, c_int) -> c_int,
    pub rms_norm: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub to_f16: unsafe extern "C" fn(G, *const f32, *mut u16, i64) -> c_int,
    pub gemm_f16: unsafe extern "C" fn(G, i64, i64, i64, *const u16, i64, *const u16, *mut f32, i64, c_int) -> c_int,
    // ink.h
    pub type_supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    pub dequant: unsafe extern "C" fn(G, c_int, P, i64, *mut f32) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(G, c_int, P, i64, *mut u16) -> c_int,
    /// (type, w, x, y, n_in, n_out, T)
    pub mv: unsafe extern "C" fn(G, c_int, P, *const f32, *mut f32, i64, i64, i64) -> c_int,
    /// (type, w, stride, scale, ids, x, per_pair, y, T, K, n_in, n_out)
    pub moe_mv: unsafe extern "C" fn(G, c_int, P, usize, *const f32, *const i32, *const f32, c_int, *mut f32, i64, i64, i64, i64) -> c_int,
    pub route: unsafe extern "C" fn(G, *const f32, *const f32, *mut i32, *mut f32, i64, i64, i64, f32) -> c_int,
    pub situ: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, *mut u16, i64, i64, f32, f32) -> c_int,
    pub combine: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, i64, i64, i64) -> c_int,
    pub combine_rows: unsafe extern "C" fn(G, *const f32, *const i32, *const i32, *const f32, *mut f32, i64, i64) -> c_int,
    pub gather_f16: unsafe extern "C" fn(G, *const f32, *const i32, *mut u16, i64, i64) -> c_int,
    pub mix: unsafe extern "C" fn(G, *const *const f32, i64, *const f32, i64, *mut f32, i64, i64, f32) -> c_int,
    pub ctrl_z: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub ctrl_apply: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, *const f32, *mut f32, i64, i64) -> c_int,
    /// (x, state, w, dyn, out, snap, T, C, H, inv)
    pub conv: unsafe extern "C" fn(G, *const f32, *mut f32, *const f32, *const f32, *mut f32, *mut f32, i64, i64, i64, f32) -> c_int,
    pub conv_keep: unsafe extern "C" fn(G, *const f32, *mut f32, i64, i64) -> c_int,
    /// (x, scale, T, H, D, thr, log_alpha, log_floor, p0)
    pub clip: unsafe extern "C" fn(G, *mut f32, *const f32, i64, i64, i64, f32, f32, f32, i64) -> c_int,
    pub rms_rows: unsafe extern "C" fn(G, *const f32, *mut f32, i64, i64, f32) -> c_int,
    pub kv_store: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, i64, i64, M, M) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    /// (q, kc, vc, r, relb, E, R, sink, out, T, Hq, Hkv, D, cap, p0, win, scale, part)
    pub attn: unsafe extern "C" fn(G, *const f32, P, P, *const f32, *const f32, i64, i64, *const f32, *mut f32, i64, i64, i64, i64, i64, i64, i64, f32,
                                   *mut f32) -> c_int,
    pub gate: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, i64) -> c_int,
    pub softcap: unsafe extern "C" fn(G, *const f32, *mut f32, i64, i64, i64, f32) -> c_int,
    pub argmax: unsafe extern "C" fn(G, *const f32, *mut i32, i64, i64) -> c_int,
    /// (logits, out, T, n, temperature, top_p, seed, pos0)
    pub sample: unsafe extern "C" fn(G, *const f32, *mut i32, i64, i64, f32, f32, u64, i64) -> c_int,
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
                // SAFETY: the symbol is declared in glm.h / ink.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            gemm: sym!("ns_gemm"),
            rms_norm: sym!("ns_rms_norm"),
            to_f16: sym!("ns_to_f16"),
            gemm_f16: sym!("ns_gemm_f16"),
            type_supported: sym!("ns_ink2_type_supported"),
            dequant: sym!("ns_ink2_dequant"),
            dequant_f16: sym!("ns_ink2_dequant_f16"),
            mv: sym!("ns_ink2_mv"),
            moe_mv: sym!("ns_ink2_moe_mv"),
            route: sym!("ns_ink2_route"),
            situ: sym!("ns_ink2_situ"),
            combine: sym!("ns_ink2_combine"),
            combine_rows: sym!("ns_ink2_combine_rows"),
            gather_f16: sym!("ns_ink2_gather_f16"),
            mix: sym!("ns_ink2_mix"),
            ctrl_z: sym!("ns_ink2_ctrl_z"),
            ctrl_apply: sym!("ns_ink2_ctrl_apply"),
            conv: sym!("ns_ink2_conv"),
            conv_keep: sym!("ns_ink2_conv_keep"),
            clip: sym!("ns_ink2_clip"),
            rms_rows: sym!("ns_ink2_rms_rows"),
            kv_store: sym!("ns_ink2_kv_store"),
            attn_scratch: sym!("ns_ink2_attn_scratch"),
            attn: sym!("ns_ink2_attn"),
            gate: sym!("ns_ink2_gate"),
            add: sym!("ns_ink2_add"),
            softcap: sym!("ns_ink2_softcap"),
            argmax: sym!("ns_ink2_argmax"),
            sample: sym!("ns_ink2_sample"),
        })
    })
    .as_ref()
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The silo's kernels (kernels/llm/inkling2/silo/silo.h, dist/silo/libnextsycl-inkling2.so), when it is there
#[allow(clippy::type_complexity)]
pub struct Silo {
    pub path: std::path::PathBuf,
    pub xq_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub xq: unsafe extern "C" fn(G, *const f32, M, i64, i64) -> c_int,
    /// (x, w, y nullable, xq, T, n, eps)
    pub rms_xq: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, M, i64, i64, f32) -> c_int,
    pub mv_supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    /// (type, w, xq, y, n_in, n_out, T)
    pub mv: unsafe extern "C" fn(G, c_int, P, P, *mut f32, i64, i64, i64) -> c_int,
    /// (n, types, ws, xqs, ys, rows, n_ins, T)
    pub mv_many: unsafe extern "C" fn(G, c_int, *const c_int, *const P, *const P, *const *mut f32, *const i64, *const i64, i64) -> c_int,
    /// (type, w, stride, scale, ids, xq, act_q, T, K, n_in, F, beta, lb)
    pub moe_gu: unsafe extern "C" fn(G, c_int, P, usize, *const f32, *const i32, P, M, i64, i64, i64, i64, f32, f32) -> c_int,
    /// (type, w, stride, scale, ids, wts, act_q, y, T, K, F, n_out)
    pub moe_down: unsafe extern "C" fn(G, c_int, P, usize, *const f32, *const i32, *const f32, P, *mut f32, i64, i64, i64, i64) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo's library (named after the engine's directory)
pub const SILO: &str = "inkling2";

/// The silo (None: this engine's own kernels for everything)
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static S: OnceLock<std::result::Result<Option<Silo>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
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
            xq_bytes: sym!("ns_ink2s_xq_bytes"),
            xq: sym!("ns_ink2s_xq"),
            rms_xq: sym!("ns_ink2s_rms_xq"),
            mv_supported: sym!("ns_ink2s_mv_supported"),
            mv: sym!("ns_ink2s_mv"),
            mv_many: sym!("ns_ink2s_mv_many"),
            moe_gu: sym!("ns_ink2s_moe_gu"),
            moe_down: sym!("ns_ink2s_moe_down"),
        }))
    })
    .as_ref()
    .map(|s| s.as_ref())
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
