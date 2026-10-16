//! This engine's own kernels (`kernels/model3d/rignet/rignet.h`), bound from the 3D kernel library by name, with checked
//! calls on device buffers (the sizes each names are the caller's to hold; the doc of each says them).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextsycl_core::{DevBuf, Error, Gpu, Result};

type G = *mut c_void;
type F = *const f32;
type O = *mut f32;
type I = *const i32;

struct Api {
    relu_affine: unsafe extern "C" fn(G, O, i64, i64, F, F) -> c_int,
    act: unsafe extern "C" fn(G, O, i64, c_int) -> c_int,
    edge_feat: unsafe extern "C" fn(G, F, i64, I, I, i64, O) -> c_int,
    seg_max: unsafe extern "C" fn(G, F, i64, I, i64, O, i64, i64) -> c_int,
    copy_cols: unsafe extern "C" fn(G, F, i64, i64, i64, O, i64, i64) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextsycl_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (the 3D library, rebuilt: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in rignet.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            relu_affine: sym!("ns_model3d_rignet_relu_affine"),
            act: sym!("ns_model3d_rignet_act"),
            edge_feat: sym!("ns_model3d_rignet_edge_feat"),
            seg_max: sym!("ns_model3d_rignet_seg_max"),
            copy_cols: sym!("ns_model3d_rignet_copy_cols"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextsycl_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("{what}: {why}")))
}

pub const TANH: i32 = 0;
pub const SIGMOID: i32 = 1;

/// The kernels on one GPU
pub struct K {
    k: &'static Api,
    g: G,
}

// SAFETY: a GPU handle (its queue is thread-safe) and function pointers.
unsafe impl Send for K {}
unsafe impl Sync for K {}

impl K {
    pub fn new(gpu: &Gpu) -> Result<K> {
        Ok(K { k: api()?, g: gpu.raw() })
    }

    /// x [m, c] = relu(x) (* scale + shift: a folded BatchNorm, both c floats)
    pub fn relu_affine(&self, x: &DevBuf, m: usize, c: usize, bn: Option<(&DevBuf, &DevBuf)>) -> Result<()> {
        if x.floats() < m * c {
            return Err(Error(format!("relu: {} floats for [{m}, {c}]", x.floats())));
        }
        let (s, t) = bn.map_or((std::ptr::null(), std::ptr::null()), |(s, t)| (s.fp() as F, t.fp() as F));
        // SAFETY: x holds m c floats (checked), the affine c each.
        check(unsafe { (self.k.relu_affine)(self.g, x.fp(), m as i64, c as i64, s, t) }, "relu")
    }

    pub fn act(&self, x: &DevBuf, n: usize, kind: i32) -> Result<()> {
        if x.floats() < n {
            return Err(Error(format!("activation: {} floats for {n}", x.floats())));
        }
        // SAFETY: x holds n floats (checked).
        check(unsafe { (self.k.act)(self.g, x.fp(), n as i64, kind) }, "activation")
    }

    /// out [e, 2c] = EdgeConv's messages of x [*, c] over the edges src -> dst (int32 buffers of e)
    pub fn edge_feat(&self, x: &DevBuf, c: usize, src: &DevBuf, dst: &DevBuf, e: usize, out: &DevBuf) -> Result<()> {
        if out.floats() < e * 2 * c || src.len < e * 4 || dst.len < e * 4 {
            return Err(Error(format!("edge features: buffers too short for {e} edges of {c}")));
        }
        // SAFETY: the edge buffers hold e int32 rows of x (the caller's graph), out e 2c floats (checked).
        check(unsafe { (self.k.edge_feat)(self.g, x.fp(), c as i64, src.ptr() as I, dst.ptr() as I, e as i64, out.fp()) }, "edge features")
    }

    /// out[i, off..off + c] (rows of `stride`) = the max of x's rows rowptr[i]..rowptr[i + 1] (n + 1 offsets)
    #[allow(clippy::too_many_arguments)]
    pub fn seg_max(&self, x: &DevBuf, c: usize, rowptr: &DevBuf, n: usize, out: &DevBuf, stride: usize, off: usize) -> Result<()> {
        if out.floats() < n.saturating_sub(1) * stride + off + c || rowptr.len < (n + 1) * 4 {
            return Err(Error(format!("segment max: buffers too short for {n} segments")));
        }
        // SAFETY: rowptr holds n + 1 offsets into x's rows (the caller's), out n rows of stride (checked).
        check(unsafe { (self.k.seg_max)(self.g, x.fp(), c as i64, rowptr.ptr() as I, n as i64, out.fp(), stride as i64, off as i64) }, "segment max")
    }

    /// dst[r, off..off + c] (rows of `dst_stride`) = src[r] (rows of `src_stride`; 0: one row to all), r < m
    #[allow(clippy::too_many_arguments)]
    pub fn copy_cols(&self, src: &DevBuf, src_stride: usize, m: usize, c: usize, dst: &DevBuf, dst_stride: usize, off: usize) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        if dst.floats() < (m - 1) * dst_stride + off + c || src.floats() < (m - 1) * src_stride + c {
            return Err(Error(format!("copy columns: buffers too short for [{m}, {c}]")));
        }
        // SAFETY: both buffers hold the rows named (checked).
        check(unsafe { (self.k.copy_cols)(self.g, src.fp(), src_stride as i64, m as i64, c as i64, dst.fp(), dst_stride as i64, off as i64) }, "copy columns")
    }
}
