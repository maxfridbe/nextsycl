//! The silo (kernels/image/vldescribe/silo/silo.h, `dist/silo/libnextsycl-vldescribe.so`): a written token's attention
//! over the cache, flash decoding. Without the file (or with NS_SILO=0) the writer uses the shared causal attention by
//! rows.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type F = *const f32;
type FM = *mut f32;

/// The silo's name (`kernels/image/vldescribe/silo` -> `dist/silo/libnextsycl-vldescribe.so`)
pub const SILO: &str = "vldescribe";

pub struct Silo {
    /// (Hq, Hkv, D)
    pub attn1_supported: unsafe extern "C" fn(i64, i64, i64) -> c_int,
    /// (n, Hq, D): the scratch attn1 needs, in floats
    pub attn1_scratch: unsafe extern "C" fn(i64, i64, i64) -> i64,
    /// (q, k, v, n, kv stride, Hq, Hkv, D, scratch, out)
    pub attn1: unsafe extern "C" fn(G, F, F, F, i64, i64, i64, i64, i64, FM, FM) -> c_int,
}

/// The silo, when built (None: NS_SILO=0, or no file)
#[allow(clippy::missing_transmute_annotations)]
pub fn silo() -> nextsycl_core::Result<Option<&'static Silo>> {
    static S: OnceLock<Result<Option<Silo>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let Some(lib) = nextsycl_core::silo(SILO).map_err(|e| e.0)? else { return Ok(None) };
        macro_rules! sym {
            ($n:literal) => {{
                let p = lib.symbol($n);
                if p.is_null() {
                    return Err(format!("{} is not in the {SILO} silo", $n));
                }
                // SAFETY: the symbol is declared in silo.h with the field's signature.
                unsafe { std::mem::transmute::<*mut c_void, _>(p) }
            }};
        }
        Ok(Some(Silo { attn1_supported: sym!("ns_image_vld_silo_attn1_supported"), attn1_scratch: sym!("ns_image_vld_silo_attn1_scratch"),
                       attn1: sym!("ns_image_vld_silo_attn1") }))
    })
    .as_ref()
    .map(Option::as_ref)
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
