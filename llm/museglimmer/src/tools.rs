//! `nextsycl llm kernels <model.gguf> [--gpu N]`: the silo's decode products timed on the model's own matrices (layer
//! 0's and the head, repacked), at the column counts a pass has - one row, and a verify pass's 2 .. 16 - as GB/s of
//! weights (NS_MG_XMX_SHAPE picks the shape: run it once a shape).

use std::path::Path;
use std::time::Instant;

use nextsycl_core::DevBuf;
use nextsycl_gguf::Gguf;

use crate::ffi;

pub fn kernels(model: &Path, gpu: usize) -> Result<(), String> {
    let e = |x: nextsycl_core::Error| x.0;
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let g = nextsycl_core::Gpu::open(gpu).map_err(e)?;
    let s = ffi::silo().map_err(e)?.ok_or("the silo is not built")?;
    println!("gpu: {}  shape {}", g.name, std::env::var("NS_MG_XMX_SHAPE").unwrap_or_else(|_| "0".into()));
    println!("{:<26} {:>14} {:>9} {:>9} {:>9} {:>9} {:>9}", "matrix", "rows x cols", "1", "2", "4", "8", "16");
    for name in ["blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.attn_q.weight", "blk.0.attn_output.weight", "blk.0.attn_k.weight", "output.weight"] {
        let t = f.tensor(name).ok_or_else(|| format!("no {name}"))?;
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        let raw = f.read(t).map_err(|x| x.0)?;
        let src = DevBuf::new(&g, raw.len()).map_err(e)?;
        src.write(0, &raw).map_err(e)?;
        // SAFETY: arithmetic only.
        let (bytes, at) = unsafe { ((s.q8r_bytes)(rows as i64, cols as i64), (s.q8r_scales_at)(rows as i64, cols as i64)) };
        let w = DevBuf::new(&g, bytes).map_err(e)?;
        // SAFETY: src holds the matrix's Q8_0 blocks, w its repacked size.
        ffi::check(unsafe { (s.q8r_pack)(g.raw(), src.ptr(), rows as i64, cols as i64, w.ptr()) }, name).map_err(e)?;
        drop(src);
        let x = DevBuf::f32(&g, 16 * cols).map_err(e)?;
        let xv: Vec<u8> = (0..16 * cols).flat_map(|i| (((i * 7919) % 1000) as f32 / 500.0 - 1.0).to_le_bytes()).collect();
        x.write(0, &xv).map_err(e)?;
        let y = DevBuf::f32(&g, 16 * rows).map_err(e)?;
        // SAFETY: arithmetic only.
        let xq = DevBuf::new(&g, unsafe { (s.xq_bytes)(cols as i64, 16) }).map_err(e)?;
        let wb = rows * cols + rows * (cols / 32) * 2;
        let mut line = format!("{name:<26} {:>14}", format!("{rows} x {cols}"));
        for n in [1usize, 2, 4, 8, 16] {
            // SAFETY: x [n, cols] -> its split form; the product into y [n, rows].
            ffi::check(unsafe { (s.xq)(g.raw(), x.fp(), std::ptr::null(), 0, xq.ptr(), cols as i64, n as i64) }, "xq").map_err(e)?;
            let run = || -> Result<(), String> {
                // SAFETY: the repacked matrix (quants, scales at `at`), x's form, y.
                ffi::check(unsafe { (s.mmvq)(g.raw(), w.ptr(), w.ptr().cast::<u8>().add(at).cast(), xq.ptr(), y.fp(), cols as i64, rows as i64, n as i64) }, "mmvq")
                    .map_err(e)
            };
            run()?;
            g.sync().map_err(e)?;
            let reps = 20;
            let t0 = Instant::now();
            for _ in 0..reps {
                run()?;
            }
            g.sync().map_err(e)?;
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            line += &format!(" {:>9.0}", wb as f64 / dt / 1e9);
        }
        println!("{line}  GB/s");
    }
    Ok(())
}
