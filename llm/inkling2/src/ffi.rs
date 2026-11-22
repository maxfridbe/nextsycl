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
    /// (inputs, x[], w[], y[], xq[], T, n, eps)
    pub rms_xq_many: unsafe extern "C" fn(G, c_int, *const *const f32, *const *const f32, *const *mut f32, *const M, i64, i64, f32) -> c_int,
    pub supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    pub packed_bytes: unsafe extern "C" fn(c_int, i64, i64) -> usize,
    /// (type, raw, rows, n_in, dst)
    pub pack: unsafe extern "C" fn(G, c_int, P, i64, i64, M) -> c_int,
    /// (type, w, rows, row0, n, n_in, dst)
    pub unpack_f16: unsafe extern "C" fn(G, c_int, P, i64, i64, i64, i64, *mut u16) -> c_int,
    /// (type, w, xq, y, n_in, n_out, T)
    pub mv: unsafe extern "C" fn(G, c_int, P, P, *mut f32, i64, i64, i64) -> c_int,
    /// (n, types, ws, xqs, ys, rows, n_ins, T)
    pub mv_many: unsafe extern "C" fn(G, c_int, *const c_int, *const P, *const P, *const *mut f32, *const i64, *const i64, i64) -> c_int,
    pub fmv_supported: unsafe extern "C" fn(c_int, i64) -> c_int,
    /// (type, w, x, y, n_in, rows, T)
    pub fmv: unsafe extern "C" fn(G, c_int, P, *const f32, *mut f32, i64, i64, i64) -> c_int,
    /// (type, w, experts, scale, ids, xq, act_q, T, K, n_in, F, beta, lb)
    pub moe_gu: unsafe extern "C" fn(G, c_int, P, i64, *const f32, *const i32, P, M, i64, i64, i64, i64, f32, f32) -> c_int,
    /// (ids, P, uni)
    pub moe_union: unsafe extern "C" fn(G, *const i32, i64, *mut i32) -> c_int,
    /// (type, w, experts, scale, uni, xq, act_q, T, K, n_in, F, beta, lb)
    pub moe_gu_union: unsafe extern "C" fn(G, c_int, P, i64, *const f32, *const i32, P, M, i64, i64, i64, i64, f32, f32) -> c_int,
    /// (type, w, experts, scale, uni, wts, act_q, part, y, T, K, F, n_out)
    pub moe_down_union: unsafe extern "C" fn(G, c_int, P, i64, *const f32, *const i32, *const f32, P, *mut f32, *mut f32, i64, i64, i64, i64) -> c_int,
    /// (type, w, experts, scale, ids, wts, act_q, y, T, K, F, n_out)
    pub moe_down: unsafe extern "C" fn(G, c_int, P, i64, *const f32, *const i32, *const f32, P, *mut f32, i64, i64, i64, i64) -> c_int,
    pub fused: Fused,
    pub draft: Draft,
    pub attn_prompt_supported: unsafe extern "C" fn(i64, i64, i64, i64) -> c_int,
    /// (q, kc, vc, r, relb, E, R, sink, out, T, Hq, Hkv, cap, p0, win)
    pub attn_prompt: unsafe extern "C" fn(G, *const f32, P, P, *const f32, *const f32, i64, i64, *const f32, *mut f32, i64, i64, i64, i64, i64, i64) -> c_int,
}

/// silo.h's Ink2sQkv: the attention's inputs after their projections (ns_ink2s_post_qkv)
#[repr(C)]
pub struct Qkv {
    pub t: i64,
    pub p0: i64,
    pub cap: i64,
    pub hq: i64,
    pub hkv: i64,
    pub d: i64,
    pub r_len: i64,
    pub kraw: *const f32,
    pub vraw: *const f32,
    pub kdyn: *const f32,
    pub vdyn: *const f32,
    pub kstate: *mut f32,
    pub vstate: *mut f32,
    pub ksnap: *mut f32,
    pub vsnap: *mut f32,
    pub kw: *const f32,
    pub vw: *const f32,
    pub inv: f32,
    pub inv_dyn: f32,
    pub kout: *mut f32,
    pub vout: *mut f32,
    pub kc: M,
    pub vc: M,
    pub q: *mut f32,
    pub r: *mut f32,
    pub qscale: *const f32,
    pub rscale: *const f32,
    pub thr: f32,
    pub rthr: f32,
    pub alpha: f32,
    pub floor: f32,
    pub eps: f32,
    pub c: i64,
    pub n0: *const f32,
    pub n1: *const f32,
    pub n2: *const f32,
    pub sinkw: *const u16,
    pub sink: *mut f32,
    pub kdw: *const u16,
    pub vdw: *const u16,
}

/// The silo's fused decode steps (dec.cpp, attn.cpp)
#[allow(clippy::type_complexity)]
pub struct Fused {
    /// (states, ns, query, nq, norm weights, out, xqs, T, C, eps_mix, eps_norm)
    pub mix_norm: unsafe extern "C" fn(G, *const *const f32, i64, *const f32, i64, *const *const f32, *mut f32, *const M, i64, i64, f32, f32) -> c_int,
    /// (x, state, w, snap, base, dst, out, T, C, inv)
    pub conv_res: unsafe extern "C" fn(G, *const f32, *mut f32, *const f32, *mut f32, *const f32, *mut f32, *mut f32, i64, i64, f32) -> c_int,
    pub post_qkv: unsafe extern "C" fn(G, *const Qkv) -> c_int,
    /// (part, sink, gate, out, xq, T, Hq, ns)
    pub merge_gate_xq: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, *mut f32, M, i64, i64, i64) -> c_int,
    /// (cur, msg, down, u, T, C, rank, eps)
    pub ctrl_down: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, *mut f32, i64, i64, i64, f32) -> c_int,
    /// (n, cur[], msg[], down[], u[], T, C, rank, eps)
    pub ctrl_down_many: unsafe extern "C" fn(G, c_int, *const *const f32, *const *const f32, *const *const f32, *const *mut f32, i64, i64, i64, f32) -> c_int,
    /// (n, cur[], msg[], u[], gain[], forget[], out[], T, C, rank)
    pub ctrl_apply_many: unsafe extern "C" fn(G, c_int, *const *const f32, *const *const f32, *const *const f32, *const *const f32, *const *const f32,
                                              *const *mut f32, i64, i64, i64) -> c_int,
    /// (cur, msg, u, gain, forget, out, T, C, rank)
    pub ctrl_apply: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, i64, i64, i64) -> c_int,
    /// (x, w, bias, logits, ids, wts, T, E, C, K, scale)
    pub router: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, *mut f32, *mut i32, *mut f32, i64, i64, i64, i64, f32) -> c_int,
    pub attn_splits: unsafe extern "C" fn(i64, i64, i64) -> i64,
    /// (q, kc, vc, r, relb, E, R, part, T, Hq, Hkv, cap, p0, win, scale)
    pub attn_part: unsafe extern "C" fn(G, *const f32, P, P, *const f32, *const f32, i64, i64, *mut f32, i64, i64, i64, i64, i64, i64, f32) -> c_int,
}

/// The silo's DSpark drafter steps (draft.cpp; attn.cpp's block attention)
#[allow(clippy::type_complexity)]
pub struct Draft {
    /// (src, feats, T, C, nf, k)
    pub feat_put: unsafe extern "C" fn(G, *const f32, *mut f32, i64, i64, i64, i64) -> c_int,
    /// (feats, nf, alpha, w, eps, y, xq, T, C)
    pub feat_norm: unsafe extern "C" fn(G, *const f32, i64, *const f32, *const f32, f32, *mut f32, M, i64, i64) -> c_int,
    /// (gate, up, xq, T, F)
    pub swiglu_xq: unsafe extern "C" fn(G, *const f32, *const f32, M, i64, i64) -> c_int,
    pub to_bf16: unsafe extern "C" fn(G, *const f32, *mut u16, i64) -> c_int,
    /// (state, tok, w1, rank, hidden, f_rnn, C, out)
    pub rnn_inputs: unsafe extern "C" fn(G, *const f32, *const i32, *const u16, i64, *const f32, *const f32, i64, *mut f32) -> c_int,
    /// (proj, bias, state, latent, rank)
    pub rnn_gates: unsafe extern "C" fn(G, *const f32, *const f32, *mut f32, *mut f32, i64) -> c_int,
    /// (base, bias, fold_b, n, cap, logits, out, prob, stop, pmin)
    pub bias_argmax: unsafe extern "C" fn(G, *const f32, *const f32, *const f32, i64, f32, *mut f32, *mut i32, *mut f32, *mut i32, f32) -> c_int,
    /// (w, x, y, rows, n, stop)
    pub fold_mv: unsafe extern "C" fn(G, *const u16, *const f32, *mut f32, i64, i64, *const i32) -> c_int,
    /// (x, T, H, w, eps, theta, p0)
    pub qk_norm_rope: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32, f32, f32, i64) -> c_int,
    /// (q, kc, vc, part, T, Hq, Hkv, cap, p0, win, scale)
    pub attn_part_block: unsafe extern "C" fn(G, *const f32, P, P, *mut f32, i64, i64, i64, i64, i64, i64, f32) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Draft {}
unsafe impl Sync for Draft {}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Fused {}
unsafe impl Sync for Fused {}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo's library (named after the engine's directory)
pub const SILO: &str = "inkling2";

/// GPU `gpu`'s silo: the variant of it that GPU matches over the default one (None: this engine's own kernels for everything)
pub fn silo_on(gpu: usize) -> nextsycl_core::Result<Option<&'static Silo>> {
    bound(nextsycl_core::silo_for(SILO, gpu)?)
}

/// A silo library's kernels, bound once a library (the default silo, or a variant over it)
fn bound(lib: Option<&'static nextsycl_core::SiloLib>) -> nextsycl_core::Result<Option<&'static Silo>> {
    static BOUND: std::sync::Mutex<Vec<(usize, &'static Silo)>> = std::sync::Mutex::new(Vec::new());
    let Some(lib) = lib else { return Ok(None) };
    let key = lib as *const nextsycl_core::SiloLib as usize;
    let mut m = BOUND.lock().unwrap();
    if let Some((_, s)) = m.iter().find(|(k, _)| *k == key) {
        return Ok(Some(*s));
    }
    let s: &'static Silo = Box::leak(Box::new(bind(lib).map_err(nextsycl_core::Error)?));
    m.push((key, s));
    Ok(Some(s))
}

fn bind(lib: &nextsycl_core::SiloLib) -> std::result::Result<Silo, String> {
    {
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
        Ok(Silo {
            path: lib.path.clone(),
            xq_bytes: sym!("ns_ink2s_xq_bytes"),
            xq: sym!("ns_ink2s_xq"),
            rms_xq: sym!("ns_ink2s_rms_xq"),
            rms_xq_many: sym!("ns_ink2s_rms_xq_many"),
            supported: sym!("ns_ink2s_supported"),
            packed_bytes: sym!("ns_ink2s_packed_bytes"),
            pack: sym!("ns_ink2s_pack"),
            unpack_f16: sym!("ns_ink2s_unpack_f16"),
            mv: sym!("ns_ink2s_mv"),
            mv_many: sym!("ns_ink2s_mv_many"),
            fmv_supported: sym!("ns_ink2s_fmv_supported"),
            fmv: sym!("ns_ink2s_fmv"),
            moe_gu: sym!("ns_ink2s_moe_gu"),
            moe_down: sym!("ns_ink2s_moe_down"),
            moe_union: sym!("ns_ink2s_moe_union"),
            moe_gu_union: sym!("ns_ink2s_moe_gu_union"),
            moe_down_union: sym!("ns_ink2s_moe_down_union"),
            attn_prompt_supported: sym!("ns_ink2s_attn_prompt_supported"),
            attn_prompt: sym!("ns_ink2s_attn_prompt"),
            draft: Draft {
                feat_put: sym!("ns_ink2s_feat_put"),
                feat_norm: sym!("ns_ink2s_feat_norm"),
                swiglu_xq: sym!("ns_ink2s_swiglu_xq"),
                to_bf16: sym!("ns_ink2s_to_bf16"),
                rnn_inputs: sym!("ns_ink2s_rnn_inputs"),
                rnn_gates: sym!("ns_ink2s_rnn_gates"),
                bias_argmax: sym!("ns_ink2s_bias_argmax"),
                fold_mv: sym!("ns_ink2s_fold_mv"),
                qk_norm_rope: sym!("ns_ink2s_qk_norm_rope"),
                attn_part_block: sym!("ns_ink2s_attn_part_block"),
            },
            fused: Fused {
                mix_norm: sym!("ns_ink2s_mix_norm"),
                conv_res: sym!("ns_ink2s_conv_res"),
                post_qkv: sym!("ns_ink2s_post_qkv"),
                merge_gate_xq: sym!("ns_ink2s_merge_gate_xq"),
                ctrl_down: sym!("ns_ink2s_ctrl_down"),
                ctrl_down_many: sym!("ns_ink2s_ctrl_down_many"),
                ctrl_apply_many: sym!("ns_ink2s_ctrl_apply_many"),
                ctrl_apply: sym!("ns_ink2s_ctrl_apply"),
                router: sym!("ns_ink2s_router"),
                attn_splits: sym!("ns_ink2s_attn_splits"),
                attn_part: sym!("ns_ink2s_attn_part"),
            },
        })
    }
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextsycl_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextsycl_core::Error(format!("{what}: {why}")))
}
