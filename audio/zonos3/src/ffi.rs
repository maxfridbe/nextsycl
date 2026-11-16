//! The kernels of the audio library this engine needs beyond the shared diffusion ones (nsd: the float32 products on
//! oneMKL, the layer norm, GELU): its own (`kernels/audio/zonos3/z3.h`) and Qwen3-TTS's per-channel scaled add
//! (`kernels/audio/qwen3tts/q3t.h`: the decoder's layer scales).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type P = *const c_void;
type F = *const f32;
type W = *mut f32;
type I = *const i32;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub embed: unsafe extern "C" fn(G, P, i64, i64, P, i64, I, i64, i64, W) -> c_int,
    pub rms: unsafe extern "C" fn(G, F, i64, i64, i64, F, f32, W) -> c_int,
    pub attnres_score: unsafe extern "C" fn(G, F, i64, i64, F, i64, f32, W) -> c_int,
    pub attnres_mix: unsafe extern "C" fn(G, F, F, i64, i64, i64, i64, i64, i64, W) -> c_int,
    pub add_vec: unsafe extern "C" fn(G, W, i64, i64, F) -> c_int,
    pub gdn_conv: unsafe extern "C" fn(G, F, i64, i64, i64, F, i64, W, W) -> c_int,
    pub gdn_prep: unsafe extern "C" fn(G, F, F, i64, i64, i64, i64, i64, F, F, W, W, W, W, W) -> c_int,
    pub gdn_scan: unsafe extern "C" fn(G, F, F, F, F, F, i64, i64, i64, W, W) -> c_int,
    pub gdn_out: unsafe extern "C" fn(G, F, F, i64, i64, i64, i64, i64, F, f32, W) -> c_int,
    pub head_rms: unsafe extern "C" fn(G, W, i64, i64, i64, i64, F, f32) -> c_int,
    pub rope: unsafe extern "C" fn(G, W, i64, i64, i64, i64, f32, i64) -> c_int,
    pub attention: unsafe extern "C" fn(G, F, i64, i64, F, F, i64, i64, i64, i64, i64, f32, i64, c_int, i64, W) -> c_int,
    pub head_gate: unsafe extern "C" fn(G, W, F, i64, i64, i64) -> c_int,
    pub softcap: unsafe extern "C" fn(G, W, i64, f32) -> c_int,
    pub silu_mul: unsafe extern "C" fn(G, F, F, W, i64) -> c_int,
    pub topk: unsafe extern "C" fn(G, F, F, i64, i64, i64, *mut i32, W) -> c_int,
    pub moe_up: unsafe extern "C" fn(G, F, i64, i64, P, P, I, i64, i64, W) -> c_int,
    pub moe_down: unsafe extern "C" fn(G, F, i64, i64, i64, P, I, F, i64, W) -> c_int,
    pub lookup: unsafe extern "C" fn(G, I, i64, i64, i64, F, i64, F, c_int, W) -> c_int,
    pub dwconv_causal: unsafe extern "C" fn(G, F, i64, i64, F, i64, F, W) -> c_int,
    pub patch_up: unsafe extern "C" fn(G, F, i64, i64, i64, W) -> c_int,
    pub spec: unsafe extern "C" fn(G, F, i64, i64, f32, W) -> c_int,
    pub ola: unsafe extern "C" fn(G, F, i64, i64, i64, F, W) -> c_int,
    pub mixer: unsafe extern "C" fn(G, F, F, i64, F, i64, W) -> c_int,
    pub resample: unsafe extern "C" fn(G, F, i64, F, i64, i64, i64, W, i64) -> c_int,
    pub conv3x3: unsafe extern "C" fn(G, F, i64, i64, i64, F, i64, i64, i64, i64, F, W) -> c_int,
    pub affine: unsafe extern "C" fn(G, F, i64, i64, F, F, c_int, F, W) -> c_int,
    pub subsample: unsafe extern "C" fn(G, F, i64, i64, i64, i64, i64, W) -> c_int,
    pub swap01: unsafe extern "C" fn(G, F, i64, i64, i64, W) -> c_int,
    pub repeat_rows: unsafe extern "C" fn(G, F, i64, i64, i64, F, W) -> c_int,
    pub softmax: unsafe extern "C" fn(G, W, i64, i64) -> c_int,
    pub mix_experts: unsafe extern "C" fn(G, F, F, i64, i64, i64, W) -> c_int,
    pub fuse4: unsafe extern "C" fn(G, F, F, F, F, F, i64, i64, W) -> c_int,
    pub astp: unsafe extern "C" fn(G, F, F, i64, i64, f32, W) -> c_int,
    pub tanh: unsafe extern "C" fn(G, W, i64) -> c_int,
    pub snake_cf: unsafe extern "C" fn(G, F, i64, i64, F, F, W) -> c_int,
    pub add: unsafe extern "C" fn(G, W, F, i64) -> c_int,
    pub patch_down: unsafe extern "C" fn(G, F, i64, i64, i64, W) -> c_int,
    pub glu: unsafe extern "C" fn(G, F, i64, i64, W) -> c_int,
    pub silu: unsafe extern "C" fn(G, W, i64) -> c_int,
    pub attn_rel: unsafe extern "C" fn(G, F, F, F, i64, i64, i64, i64, F, i64, i64, F, f32, c_int, W) -> c_int,
    pub bf16: unsafe extern "C" fn(G, W, i64) -> c_int,
    pub add_rows: unsafe extern "C" fn(G, W, F, i64, i64) -> c_int,
    pub moe_lora_t: unsafe extern "C" fn(G, F, i64, i64, I, i64, i64, F, i64, W) -> c_int,
    pub moe_up_lora: unsafe extern "C" fn(G, F, i64, i64, P, P, I, i64, i64, F, F, F, F, i64, W) -> c_int,
    pub moe_down_lora: unsafe extern "C" fn(G, F, i64, i64, i64, P, I, F, i64, F, F, i64, W) -> c_int,
    pub attn_full: unsafe extern "C" fn(G, F, F, F, i64, i64, i64, i64, f32, W) -> c_int,
    pub rope_tab: unsafe extern "C" fn(G, W, i64, i64, i64, i64, F, F) -> c_int,
    pub col2im: unsafe extern "C" fn(G, F, i64, i64, i64, i64, i64, F, i64, W) -> c_int,
    pub attn_window: unsafe extern "C" fn(G, F, F, F, i64, i64, i64, i64, i64, f32, *const i32, i64, i64, i64, W) -> c_int,
    pub dwconv_centred: unsafe extern "C" fn(G, F, i64, i64, F, i64, F, W) -> c_int,
    /// q3t's: x [R, C] += scale[c] y [R, C]
    pub scale_add: unsafe extern "C" fn(G, W, F, F, i64, i64) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the audio library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in z3.h / q3t.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            embed: sym!("ns_audio_z3_embed"),
            rms: sym!("ns_audio_z3_rms"),
            attnres_score: sym!("ns_audio_z3_attnres_score"),
            attnres_mix: sym!("ns_audio_z3_attnres_mix"),
            add_vec: sym!("ns_audio_z3_add_vec"),
            gdn_conv: sym!("ns_audio_z3_gdn_conv"),
            gdn_prep: sym!("ns_audio_z3_gdn_prep"),
            gdn_scan: sym!("ns_audio_z3_gdn_scan"),
            gdn_out: sym!("ns_audio_z3_gdn_out"),
            head_rms: sym!("ns_audio_z3_head_rms"),
            rope: sym!("ns_audio_z3_rope"),
            attention: sym!("ns_audio_z3_attention"),
            head_gate: sym!("ns_audio_z3_head_gate"),
            softcap: sym!("ns_audio_z3_softcap"),
            silu_mul: sym!("ns_audio_z3_silu_mul"),
            topk: sym!("ns_audio_z3_topk"),
            moe_up: sym!("ns_audio_z3_moe_up"),
            moe_down: sym!("ns_audio_z3_moe_down"),
            lookup: sym!("ns_audio_z3_lookup"),
            dwconv_causal: sym!("ns_audio_z3_dwconv_causal"),
            patch_up: sym!("ns_audio_z3_patch_up"),
            spec: sym!("ns_audio_z3_spec"),
            ola: sym!("ns_audio_z3_ola"),
            mixer: sym!("ns_audio_z3_mixer"),
            resample: sym!("ns_audio_z3_resample"),
            conv3x3: sym!("ns_audio_z3_conv3x3"),
            affine: sym!("ns_audio_z3_affine"),
            subsample: sym!("ns_audio_z3_subsample"),
            swap01: sym!("ns_audio_z3_swap01"),
            repeat_rows: sym!("ns_audio_z3_repeat_rows"),
            softmax: sym!("ns_audio_z3_softmax"),
            mix_experts: sym!("ns_audio_z3_mix_experts"),
            fuse4: sym!("ns_audio_z3_fuse4"),
            astp: sym!("ns_audio_z3_astp"),
            tanh: sym!("ns_audio_z3_tanh"),
            snake_cf: sym!("ns_audio_z3_snake_cf"),
            add: sym!("ns_audio_z3_add"),
            patch_down: sym!("ns_audio_z3_patch_down"),
            glu: sym!("ns_audio_z3_glu"),
            silu: sym!("ns_audio_z3_silu"),
            attn_rel: sym!("ns_audio_z3_attn_rel"),
            bf16: sym!("ns_audio_z3_bf16"),
            add_rows: sym!("ns_audio_z3_add_rows"),
            rope_tab: sym!("ns_audio_z3_rope_tab"),
            attn_full: sym!("ns_audio_z3_attn_full"),
            moe_lora_t: sym!("ns_audio_z3_moe_lora_t"),
            moe_up_lora: sym!("ns_audio_z3_moe_up_lora"),
            moe_down_lora: sym!("ns_audio_z3_moe_down_lora"),
            col2im: sym!("ns_audio_z3_col2im"),
            attn_window: sym!("ns_audio_z3_attn_window"),
            dwconv_centred: sym!("ns_audio_z3_dwconv_same"),
            scale_add: sym!("ns_audio_q3t_scale_add"),
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
