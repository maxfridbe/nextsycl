//! The silo (kernels/image/h3image/silo/silo.h, `dist/silo/libnextsycl-h3image.so`: P-H3-Image's silo copied - the same
//! H3 denoiser attention): the long-sequence attention with its own preparation passes, and the GGUF block matrices'
//! expansion to bfloat16 (Q3_K read as stored). Without the file (or with
//! NS_SILO=0) the blocks use the shared kernels'.

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use nextsycl_core::Gpu;
use nextsycl_h3::device::Tensor;
use nextsycl_h3::dit::{AttentionFn, ExpandFn};
use nextsycl_h3::dtype::DType;
use nextsycl_h3::gguf::GType;
use nextsycl_h3::ops::KLinear;
use nextsycl_h3::Error;

/// The silo's name (`kernels/image/h3image/silo` -> `dist/silo/libnextsycl-h3image.so`)
pub const SILO: &str = "h3image";

#[allow(clippy::type_complexity)]
pub struct Silo {
    pub sage_supported: unsafe extern "C" fn(*mut c_void, i64, i64, i64) -> i32,
    pub sage: unsafe extern "C" fn(*mut c_void, *const c_void, i64, i64, i64, i64, *mut c_void) -> i32,
    pub expand_supported: unsafe extern "C" fn(i32) -> i32,
    pub expand: unsafe extern "C" fn(*mut c_void, *const c_void, i32, i64, *mut c_void) -> i32,
}

/// The GGUF type code the silo's expansion takes for `kind` (None: not one it reads)
pub fn gguf_code(kind: GType) -> Option<i32> {
    match kind {
        GType::Q8_0 => Some(8),
        GType::Q3K => Some(11),
        GType::Q4K => Some(12),
        GType::Q5K => Some(13),
        _ => None,
    }
}

/// Whether the silo expands `kind`
pub fn expands(s: &Silo, kind: GType) -> bool {
    // SAFETY: plain query.
    gguf_code(kind).is_some_and(|c| unsafe { (s.expand_supported)(c) } != 0)
}

/// `n` values of `kind` blocks (`blocks`, device bytes) into the bfloat16 buffer `out` with the silo's expansion
pub fn expand_into(gpu: &Gpu, s: &Silo, blocks: &Tensor, kind: GType, n: usize, out: &Tensor) -> nextsycl_h3::Result<()> {
    let code = gguf_code(kind).ok_or_else(|| Error(format!("the silo does not expand {kind:?}")))?;
    if out.dtype != DType::BF16 || out.elements() < n || blocks.buf.len() < kind.bytes(n) {
        return Err(Error(format!("silo expand: {} bytes of {kind:?} for {n} values into {:?} {:?}", blocks.buf.len(), out.dtype, out.shape)));
    }
    // SAFETY: device buffers on the GPU's queue, sizes checked above.
    let rc = unsafe { (s.expand)(gpu.raw(), blocks.buf.ptr().cast_const(), code, n as i64, out.buf.ptr()) };
    if rc != 0 {
        return Err(Error(format!("the silo's expansion: {}", nextsycl_core::api().map(|a| a.error()).unwrap_or_default())));
    }
    Ok(())
}

/// The blocks' k-quant expansion on the silo for the kinds it reads (the rest, and a 16-bit buffer other than
/// bfloat16, stay on the shared kernels'); the engine's device must share `gpu`'s queue (`Device::on_queue`)
pub fn expander(gpu: &Arc<Gpu>, s: &'static Silo) -> ExpandFn {
    let gpu = gpu.clone();
    Box::new(move |l: &KLinear, w: &Tensor| {
        if w.dtype != DType::BF16 || !expands(s, l.kind) {
            return Ok(false);
        }
        expand_into(&gpu, s, &l.blocks, l.kind, l.n * l.k, w)?;
        Ok(true)
    })
}

/// The silo, when built (None: NS_SILO=0, or no file - the shared kernels throughout)
// each field's declared type is the annotation: the transmute target is inferred from it
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
        Ok(Some(Silo {
            sage_supported: sym!("ns_image_h3i_silo_sage_supported"),
            sage: sym!("ns_image_h3i_silo_sage"),
            expand_supported: sym!("ns_image_h3i_silo_expand_supported"),
            expand: sym!("ns_image_h3i_silo_expand"),
        }))
    })
    .as_ref()
    .map(Option::as_ref)
    .map_err(|e| nextsycl_core::Error(e.clone()))
}

/// The blocks' attention on the silo, for the denoiser's packed bfloat16 qkv rows (q, k, v side by side), from `min_s`
/// tokens (shorter ones stay on the shared kernels' oneDNN attention) where it takes the shape; the engine's device must
/// share `gpu`'s queue (`Device::on_queue`)
pub fn attention(gpu: &Arc<Gpu>, s: &'static Silo, min_s: usize) -> AttentionFn {
    let gpu = gpu.clone();
    Box::new(move |q, k, v, out| {
        let (t, h, d) = (q.tokens, q.heads, q.dim);
        if t < min_s {
            return Ok(false);
        }
        let hd = h * d;
        if q.t.dtype != DType::BF16 || out.dtype != DType::BF16 || q.offset != 0 || k.offset != hd || v.offset != 2 * hd || out.elements() != t * hd
            || q.t.elements() < t * q.stride
        {
            return Ok(false);
        }
        // SAFETY: plain query of sizes.
        if unsafe { (s.sage_supported)(gpu.raw(), t as i64, h as i64, d as i64) } == 0 {
            return Ok(false);
        }
        // SAFETY: the qkv buffer holds t rows of `stride` bfloat16 (checked), out t x hd bfloat16; one queue with the device.
        let rc = unsafe { (s.sage)(gpu.raw(), q.t.buf.ptr(), t as i64, h as i64, d as i64, q.stride as i64, out.buf.ptr()) };
        if rc != 0 {
            return Err(Error(format!("the silo's attention: {}", nextsycl_core::api().map(|a| a.error()).unwrap_or_default())));
        }
        Ok(true)
    })
}
