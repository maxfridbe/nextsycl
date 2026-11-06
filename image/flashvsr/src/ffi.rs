//! This engine's own kernels (`kernels/image/flashvsr/fv.h`), bound from the image kernel library by name, and its silo
//! (`kernels/image/flashvsr/silo/silo.h`: the attention on sycl-tla, `dist/silo/libnextsycl-flashvsr.so`).

use std::ffi::{c_char, c_int, c_void};
use std::sync::{Arc, OnceLock};

use nextsycl_core::Gpu;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    // the DiT's tokens
    pub patchify: unsafe extern "C" fn(G, *const f32, i64, i64, i64, M) -> c_int,
    pub unpatchify_sub: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, *mut f32) -> c_int,
    pub add_lq: unsafe extern "C" fn(G, *mut f32, *const f32, i64, i64, i64, i64) -> c_int,
    pub ln: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const f32, *const f32, M) -> c_int,
    pub gate_add: unsafe extern "C" fn(G, *mut f32, *const f32, i64, i64, *const f32) -> c_int,
    pub rms_rope: unsafe extern "C" fn(G, M, i64, i64, i64, *const f32, f32, *const f32, *const f32, i64, i64) -> c_int,
    // the draft and the plain attention
    pub block_mean: unsafe extern "C" fn(G, P, i64, i64, i64, *mut f32) -> c_int,
    pub block_probs: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, i64, i64, i64, *const i32, *mut f32) -> c_int,
    pub topk_thr: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32) -> c_int,
    pub block_lists: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, i64, c_int, i64, *mut i32, *mut i32) -> c_int,
    pub zero_empty: unsafe extern "C" fn(G, M, i64, i64, i64, *const i32) -> c_int,
    pub sparse_attn: unsafe extern "C" fn(G, P, i64, i64, P, P, i64, *const i32, *const i32, i64, M) -> c_int,
    // the LQ projection
    pub unshuffle: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, M) -> c_int,
    pub stack_taps: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, M) -> c_int,
    pub rms_silu: unsafe extern "C" fn(G, M, i64, i64, *const f32) -> c_int,
    // the decoder
    pub tcd_input: unsafe extern "C" fn(G, P, *const f32, i64, i64, i64, i64, i64, M) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, i64, M) -> c_int,
    // frames, colour, noise
    pub u8_to_half: unsafe extern "C" fn(G, *const u8, i64, M) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, i64, M) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, i64, *mut f32) -> c_int,
    pub adain: unsafe extern "C" fn(G, P, P, i64, i64, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    pub to_u8: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, *const f32, *mut u8) -> c_int,
    pub randn: unsafe extern "C" fn(G, *mut f32, i64, u64, u64) -> c_int,
    // products (oneDNN)
    pub dnn_clear: unsafe extern "C" fn(G) -> c_int,
    pub gemm: unsafe extern "C" fn(G, P, i64, i64, i64, P, i64, *const f32, c_int, *const f32, c_int, M, i64, c_int) -> c_int,
    pub conv: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, P, i64, i64, i64, *const f32, c_int, c_int, M) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the image library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in fv.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            patchify: sym!("ns_image_flashvsr_patchify"),
            unpatchify_sub: sym!("ns_image_flashvsr_unpatchify_sub"),
            add_lq: sym!("ns_image_flashvsr_add_lq"),
            ln: sym!("ns_image_flashvsr_ln"),
            gate_add: sym!("ns_image_flashvsr_gate_add"),
            rms_rope: sym!("ns_image_flashvsr_rms_rope"),
            block_mean: sym!("ns_image_flashvsr_block_mean"),
            block_probs: sym!("ns_image_flashvsr_block_probs"),
            topk_thr: sym!("ns_image_flashvsr_topk_thr"),
            block_lists: sym!("ns_image_flashvsr_block_lists"),
            zero_empty: sym!("ns_image_flashvsr_zero_empty"),
            sparse_attn: sym!("ns_image_flashvsr_sparse_attn"),
            unshuffle: sym!("ns_image_flashvsr_unshuffle"),
            stack_taps: sym!("ns_image_flashvsr_stack_taps"),
            rms_silu: sym!("ns_image_flashvsr_rms_silu"),
            tcd_input: sym!("ns_image_flashvsr_tcd_input"),
            up2: sym!("ns_image_flashvsr_up2"),
            u8_to_half: sym!("ns_image_flashvsr_u8_to_half"),
            to_half: sym!("ns_image_flashvsr_to_half"),
            to_float: sym!("ns_image_flashvsr_to_float"),
            adain: sym!("ns_image_flashvsr_adain"),
            to_u8: sym!("ns_image_flashvsr_to_u8"),
            randn: sym!("ns_image_flashvsr_randn"),
            dnn_clear: sym!("ns_image_flashvsr_dnn_clear"),
            gemm: sym!("ns_image_flashvsr_gemm"),
            conv: sym!("ns_image_flashvsr_conv"),
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

/// The kernels' handle of a GPU (`ns_gpu*`)
pub fn g(gpu: &Arc<Gpu>) -> G {
    gpu.raw()
}

/// The silo's name (`kernels/image/flashvsr/silo` -> `dist/silo/libnextsycl-flashvsr.so`)
pub const SILO: &str = "flashvsr";

/// The silo's kernels (silo.h): they take the GPU's SYCL queue (`Gpu::queue`)
pub struct Silo {
    pub error: unsafe extern "C" fn() -> *const c_char,
    pub supported: unsafe extern "C" fn(M) -> c_int,
    pub sparse: unsafe extern "C" fn(M, P, P, P, M, i64, i64, i64, i64, *const i32, *const i32, i64, i64) -> c_int,
    pub attention: unsafe extern "C" fn(M, P, P, P, M, i64, i64, i64, i64, i64, i64) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

impl Silo {
    /// A silo call's return code as a result (its own reason)
    pub fn check(&self, rc: c_int, what: &str) -> nextsycl_core::Result<()> {
        if rc == 0 {
            return Ok(());
        }
        // SAFETY: a NUL-terminated string the silo keeps (its thread's last error).
        let why = unsafe { std::ffi::CStr::from_ptr((self.error)()) }.to_string_lossy().into_owned();
        Err(nextsycl_core::Error(format!("{what}: {why}")))
    }
}

/// The silo, when built (None: NS_SILO=0, or no file - the engine's own kernels throughout)
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
        Ok(Some(Silo { error: sym!("ns_image_flashvsr_silo_error"), supported: sym!("ns_image_flashvsr_silo_supported"),
                        sparse: sym!("ns_image_flashvsr_silo_sparse"), attention: sym!("ns_image_flashvsr_silo_attention") }))
    })
    .as_ref()
    .map(|s| s.as_ref())
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The silo for this GPU, when it has one and it runs there
pub fn silo_on(gpu: &Gpu) -> nextsycl_core::Result<Option<&'static Silo>> {
    Ok(match silo()? {
        // SAFETY: the GPU's live queue.
        Some(s) if unsafe { (s.supported)(gpu.queue()) } == 1 => Some(s),
        _ => None,
    })
}

/// IEEE half from float32, rounded to nearest even (the weights made half at load)
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

/// float32 values as half bytes
pub fn half_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect()
}

/// float32 values as their bytes
pub fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: plain floats viewed as their bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_to_nearest_even() {
        let back = |x: f32| nextsycl_gguf::safetensors::f16_to_f32(f32_to_f16(x));
        for x in [0.0f32, 1.0, -2.5, 0.333_333, 65504.0, 6.1e-5, 1e-7, -1e-6] {
            let e = (back(x) - x).abs();
            assert!(e <= x.abs() * 1e-3 + 1e-7, "{x} -> {}", back(x));
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(70000.0), 0x7c00);
        // 1 + 2^-11 is halfway between 1 and the next half: to even (1.0)
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
    }
}
