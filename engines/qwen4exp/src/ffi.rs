//! This engine's own C ABI (`kernels/engines/qwen4exp/qwen.h`), bound from the kernel library by name. The structs
//! mirror the header field for field.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

/// `ns_qw_layer`: one layer's weights (device pointers, null for the kind it is not)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Layer {
    pub hc_norm: [*const f32; 2],
    pub hc_down: [*const u16; 2],
    pub hc_up: [*const u16; 2],
    pub hc_inject: [*const u16; 2],
    pub qkv_type: c_int,
    pub z_type: c_int,
    pub out_type: c_int,
    pub qkv: *const c_void,
    pub z_w: *const c_void,
    pub out_w: *const c_void,
    pub alpha: *const u16,
    pub beta: *const u16,
    pub conv: *const f32,
    pub ssm_a: *const f32,
    pub dt_bias: *const f32,
    pub ssm_norm: *const f32,
    pub q_type: c_int,
    pub k_type: c_int,
    pub v_type: c_int,
    pub o_type: c_int,
    pub q_w: *const c_void,
    pub k_w: *const c_void,
    pub v_w: *const c_void,
    pub o_w: *const c_void,
    pub q_norm: *const f32,
    pub k_norm: *const f32,
    pub idx_q: *const u16,
    pub idx_k: *const u16,
    pub idx_q_norm: *const f32,
    pub idx_k_norm: *const f32,
    pub router: *const u16,
    pub sh_gate_inp: *const u16,
    pub sh_gate_type: c_int,
    pub sh_up_type: c_int,
    pub sh_down_type: c_int,
    pub sh_gate: *const c_void,
    pub sh_up: *const c_void,
    pub sh_down: *const c_void,
    pub gu_type: c_int,
    pub d_type: c_int,
    pub ple_key: *const u16,
    pub ple_value: *const u16,
    pub ple_norm_key: *const f32,
    pub ple_norm_query: *const f32,
    pub ple_norm_conv: *const f32,
    pub ple_conv: *const u16,
}

impl Default for Layer {
    fn default() -> Layer {
        // SAFETY: every field is an integer or a raw pointer, for which all-zero is valid (0 / null).
        unsafe { std::mem::zeroed() }
    }
}

/// `ns_qw_edges`: the embedding (first stage), the final mixer and head (last stage)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Edges {
    pub embd_type: c_int,
    pub embd: *const c_void,
    pub embd_row: usize,
    pub out_hc_norm: *const f32,
    pub out_hc_down: *const u16,
    pub out_hc_up: *const u16,
    pub out_type: c_int,
    pub out: *const c_void,
    pub vocab: i64,
}

impl Default for Edges {
    fn default() -> Edges {
        // SAFETY: integers and raw pointers only.
        unsafe { std::mem::zeroed() }
    }
}

/// `ns_qw_desc`
#[repr(C)]
pub struct Desc {
    pub lb: i64,
    pub le: i64,
    pub n_layer: i64,
    pub n_expert: i64,
    pub max_cells: i64,
    pub layers: *const Layer,
    pub edges: Edges,
    pub d_res: *const i32,
    pub cache_base: *const u8,
    pub slot_off: *const u64,
    pub n_slots: i64,
    pub h_res: *const i32,
    pub mirror: *const u64,
    pub h_mirror: *const u64,
}

pub type Stage = *mut c_void;
pub type State = *mut c_void;
type G = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub new: unsafe extern "C" fn(G, *const Desc, *mut Stage) -> c_int,
    pub free: unsafe extern "C" fn(Stage),
    pub buffers: unsafe extern "C" fn(Stage, *mut *mut f32, *mut *mut f32, *mut usize) -> c_int,
    pub state_new: unsafe extern "C" fn(Stage, i64, *mut State) -> c_int,
    pub state_free: unsafe extern "C" fn(State),
    pub state_reset: unsafe extern "C" fn(Stage, State) -> c_int,
    pub state_warm: unsafe extern "C" fn(Stage, State, c_int) -> c_int,
    pub state_bytes: unsafe extern "C" fn(State, i64, *mut u64) -> c_int,
    pub state_save: unsafe extern "C" fn(Stage, State, i64, *mut c_void) -> c_int,
    pub state_load: unsafe extern "C" fn(Stage, State, i64, *const c_void) -> c_int,
    pub state_copy: unsafe extern "C" fn(Stage, State, State, i64) -> c_int,
    pub window: unsafe extern "C" fn(Stage, State, c_int, *const i32, i64, *const f32, c_int, *mut f32, *mut i32) -> c_int,
    pub commit: unsafe extern "C" fn(Stage, State, c_int) -> c_int,
    pub prefill_buffers: unsafe extern "C" fn(Stage, i64, *mut *mut f32) -> c_int,
    pub prefill: unsafe extern "C" fn(Stage, State, i64, *const i32, i64, *const f32) -> c_int,
    pub cvec_set: unsafe extern "C" fn(Stage, *const f32, *const f32, c_int, c_int) -> c_int,
    pub mtp_load: unsafe extern "C" fn(Stage, *const std::ffi::c_char, c_int, i64, c_int, *const c_void, usize) -> c_int,
    pub mtp_prefill: unsafe extern "C" fn(Stage, State, i64, i64, *const i32) -> c_int,
    pub mtp_draft: unsafe extern "C" fn(Stage, State, *const i32, i64, c_int, c_int, f32, *mut i32, *mut f32, *mut c_int) -> c_int,
    /// Strata's grouped expert kernels take these gate/up and down formats (glm.h's ns_moe_grouped_supported)
    pub moe_grouped_supported: unsafe extern "C" fn(c_int, c_int, i64, i64) -> c_int,
}

// SAFETY: plain function pointers into the library, which stays loaded.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

/// The engine's functions, bound once
pub fn api() -> ns_core::Result<&'static Api> {
    API.get_or_init(|| {
        let lib = ns_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($n:literal) => {{
                let p = lib.symbol($n);
                if p.is_null() {
                    return Err(format!("the kernel library has no {} (built without kernels/engines/qwen4exp?)", $n));
                }
                // SAFETY: the symbol is declared in qwen.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            new: sym!("ns_qw_new"),
            free: sym!("ns_qw_free"),
            buffers: sym!("ns_qw_buffers"),
            state_new: sym!("ns_qw_state_new"),
            state_free: sym!("ns_qw_state_free"),
            state_reset: sym!("ns_qw_state_reset"),
            state_warm: sym!("ns_qw_state_warm"),
            state_bytes: sym!("ns_qw_state_bytes"),
            state_save: sym!("ns_qw_state_save"),
            state_load: sym!("ns_qw_state_load"),
            state_copy: sym!("ns_qw_state_copy"),
            window: sym!("ns_qw_window"),
            commit: sym!("ns_qw_commit"),
            prefill_buffers: sym!("ns_qw_prefill_buffers"),
            prefill: sym!("ns_qw_prefill"),
            cvec_set: sym!("ns_qw_cvec_set"),
            mtp_load: sym!("ns_qw_mtp_load"),
            mtp_prefill: sym!("ns_qw_mtp_prefill"),
            mtp_draft: sym!("ns_qw_mtp_draft"),
            moe_grouped_supported: sym!("ns_moe_grouped_supported"),
        })
    })
    .as_ref()
    .map_err(|e| ns_core::Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> ns_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = ns_core::api().map(|a| a.error()).unwrap_or_default();
    Err(ns_core::Error(format!("{what}: {why}")))
}
