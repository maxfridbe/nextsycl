//! The submanifold 3 x 3 x 3 convolution (spconv's SubMConv3d) as gather + GEMM: each voxel's 27 neighbours' features
//! gathered into a row (tap-major, missing ones zero), a row block at a time, times the weight [Co, 27 Ci] (spconv's
//! [Co, 3, 3, 3, Ci] as stored). The trellis2 engine's, copied.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

use crate::w::{File, Lin};
use crate::Ctx;

/// bytes a row block of the gather may take
const GATHER_BYTES: usize = 384 << 20;

pub struct SConv {
    pub lin: Lin,
    pub ci: usize,
    pub co: usize,
}

impl SConv {
    /// `p.weight` [Co, 3, 3, 3, Ci] and its bias
    pub fn load(f: &File, gpu: &Arc<Gpu>, p: &str) -> Result<SConv> {
        let (sh, _, _) = f.raw(&format!("{p}.weight"))?;
        if sh.len() != 5 || sh[1..4] != [3, 3, 3] {
            return Err(Error(format!("{p}.weight: {sh:?}, not [Co, 3, 3, 3, Ci]")));
        }
        Ok(SConv { lin: Lin::load(f, gpu, p, false)?, ci: sh[4], co: sh[0] })
    }

    /// out [n, co] (out_dt) = the convolution of x (half [n, ci]) over the neighbour map `nmap` (int32 [n, 27])
    pub fn apply(&self, cx: &Ctx, x: &DevBuf, n: usize, nmap: &DevBuf, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let row = 27 * self.ci * 2;
        let rows = (GATHER_BYTES / row).clamp(1, n);
        let gbuf = DevBuf::new(&cx.gpu, rows * row)?;
        let es = if out_dt == Dt::F32 { 4 } else { 2 };
        for r0 in (0..n).step_by(rows) {
            let m = rows.min(n - r0);
            cx.k.gather27(x.ptr(), self.ci, nmap.ptr().cast(), r0, m, gbuf.ptr())?;
            // SAFETY: rows r0 .. r0 + m of out's n rows of co values.
            let o = unsafe { out.cast::<u8>().add(r0 * self.co * es) }.cast();
            self.lin.apply(cx, gbuf.ptr(), m, o, out_dt)?;
        }
        Ok(())
    }
}
