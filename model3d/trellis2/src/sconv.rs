//! The submanifold 3 x 3 x 3 convolution as gather + GEMM: each voxel's 27 neighbours' features gathered into a row
//! (tap-major, missing ones zero), a row block at a time, times the weight [Co, 27 Ci] (flex_gemm's [Co, Kd, Kh, Kw, Ci]
//! as stored). A dense grid is the voxel set of all its cells (the sparse-structure decoder's zero padding is the
//! missing neighbours).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

use crate::w::{File, Keep, Lin};
use crate::Ctx;

/// bytes a row block of the gather may take
const GATHER_BYTES: usize = 384 << 20;

pub struct SConv {
    pub lin: Lin,
    pub ci: usize,
    pub co: usize,
}

impl SConv {
    /// `p.weight` [Co, 3, 3, 3, Ci] (the sparse decoders') and its bias
    pub fn load(f: &File, gpu: &Arc<Gpu>, p: &str) -> Result<SConv> {
        let (sh, _, _) = f.raw(&format!("{p}.weight"))?;
        if sh.len() != 5 || sh[1..4] != [3, 3, 3] {
            return Err(Error(format!("{p}.weight: {sh:?}, not [Co, 3, 3, 3, Ci]")));
        }
        Ok(SConv { lin: Lin::load(f, gpu, p, Keep::Half)?, ci: sh[4], co: sh[0] })
    }

    /// `p.weight` [Co, Ci, 3, 3, 3] (a dense Conv3d's), reordered to [Co, 27, Ci]
    pub fn load_dense(f: &File, gpu: &Arc<Gpu>, p: &str) -> Result<SConv> {
        let (sh, w) = f.host(&format!("{p}.weight"))?;
        if sh.len() != 5 || sh[2..] != [3, 3, 3] {
            return Err(Error(format!("{p}.weight: {sh:?}, not [Co, Ci, 3, 3, 3]")));
        }
        let (co, ci) = (sh[0], sh[1]);
        let mut r = vec![0f32; w.len()];
        for o in 0..co {
            for c in 0..ci {
                for t in 0..27 {
                    r[(o * 27 + t) * ci + c] = w[(o * ci + c) * 27 + t];
                }
            }
        }
        let b = f.vec(&format!("{p}.bias"))?;
        Ok(SConv { lin: Lin::from_host(gpu, &r, Some(&b), co, 27 * ci, Keep::Half)?, ci, co })
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
            self.lin.apply(&cx.nsd, &cx.k, &gbuf, gbuf.ptr(), m, o, out_dt)?;
        }
        Ok(())
    }
}

/// A neighbour map on the GPU
pub fn upload_nmap(gpu: &Arc<Gpu>, nmap: &[i32]) -> Result<DevBuf> {
    let b = DevBuf::new(gpu, nmap.len() * 4)?;
    // SAFETY: the i32 values' own bytes (the GPU reads them little-endian, as the host holds them).
    b.write(0, unsafe { std::slice::from_raw_parts(nmap.as_ptr().cast::<u8>(), nmap.len() * 4) })?;
    Ok(b)
}

/// The cells of a dense d^3 grid in row-major (x, y, z) order
pub fn grid(d: usize) -> Vec<[i32; 3]> {
    (0..d * d * d).map(|i| [(i / (d * d)) as i32, (i / d % d) as i32, (i % d) as i32]).collect()
}
