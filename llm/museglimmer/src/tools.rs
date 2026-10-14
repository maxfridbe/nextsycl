//! `nextsycl llm kernels <model.gguf> [--gpu N]`: the silo's decode products timed on the model's own matrices (layer
//! 0's and the head, repacked), at the column counts a pass has - one row, and a verify pass's 2 .. 16 - as GB/s of
//! weights (NS_MG_XMX_SHAPE picks Q8_0's shape, NS_MG_KQ_SHAPE the K-quants': run it once a shape). A K-quant's
//! product is checked too: against the dequantized matrix times the quantized columns on the host (its first rows),
//! and a 16-column pass's first column against the one-column pass's (bit for bit).

use std::os::raw::c_int;
use std::path::Path;
use std::time::Instant;

use nextsycl_core::DevBuf;
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;

fn f16(b: u16) -> f32 {
    let (e, m) = (((b >> 10) & 31) as i32, (b & 1023) as f32);
    let v = match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    };
    if b >> 15 == 1 { -v } else { v }
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn kernels(model: &Path, gpu: usize) -> Result<(), String> {
    let e = |x: nextsycl_core::Error| x.0;
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let g = nextsycl_core::Gpu::open(gpu).map_err(e)?;
    let k = ffi::api().map_err(e)?;
    let s = ffi::silo().map_err(e)?.ok_or("the silo is not built")?;
    println!("gpu: {}  shape {} (Q8_0) / {} (K-quants)", g.name, std::env::var("NS_MG_XMX_SHAPE").unwrap_or_else(|_| "7".into()),
             std::env::var("NS_MG_KQ_SHAPE").unwrap_or_else(|_| "0".into()));
    println!("{:<26} {:>5} {:>14} {:>9} {:>9} {:>9} {:>9} {:>9}", "matrix", "type", "rows x cols", "1", "2", "4", "8", "16");
    let mut names: Vec<String> = ["blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.attn_q.weight", "blk.0.attn_output.weight", "blk.0.attn_k.weight",
                                  "blk.0.attn_v.weight", "output.weight"].iter().map(|x| x.to_string()).collect();
    // a layer's matrices of another type (Q4_K_M: some ffn_down / attn_v Q6_K)
    for l in 1..f.tensors.len() {
        let n = format!("blk.{l}.ffn_down.weight");
        match (f.tensor(&n), f.tensor("blk.0.ffn_down.weight")) {
            (Some(t), Some(t0)) if t.ty != t0.ty => {
                names.push(n);
                break;
            }
            (None, _) => break,
            _ => {}
        }
    }
    for name in &names {
        let Some(t) = f.tensor(name) else { continue };
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        let raw = f.read(t).map_err(|x| x.0)?;
        let src = DevBuf::new(&g, raw.len()).map_err(e)?;
        src.write(0, &raw).map_err(e)?;
        let ty = t.ty.code() as c_int;
        // SAFETY: arithmetic only.
        let kq = unsafe { (s.kr_supported)(ty, cols as i64) } != 0;
        let (w, at, wb) = if t.ty == GType::Q8_0 {
            // SAFETY: arithmetic only.
            let (bytes, at) = unsafe { ((s.q8r_bytes)(rows as i64, cols as i64), (s.q8r_scales_at)(rows as i64, cols as i64)) };
            let w = DevBuf::new(&g, bytes).map_err(e)?;
            // SAFETY: src holds the matrix's Q8_0 blocks, w its repacked size.
            ffi::check(unsafe { (s.q8r_pack)(g.raw(), src.ptr(), rows as i64, cols as i64, w.ptr()) }, name).map_err(e)?;
            (w, at, rows * cols + rows * (cols / 32) * 2)
        } else if kq {
            // SAFETY: arithmetic only.
            let rb = unsafe { (s.kr_row_bytes)(ty, cols as i64) };
            let w = DevBuf::new(&g, rows * rb).map_err(e)?;
            // SAFETY: src holds the matrix's blocks, w its repacked rows.
            ffi::check(unsafe { (s.kr_pack)(g.raw(), ty, src.ptr(), rows as i64, cols as i64, w.ptr()) }, name).map_err(e)?;
            (w, 0, raw.len())
        } else {
            println!("{name:<26} {:>5?}  not the silo's", t.ty);
            continue;
        };
        let x = DevBuf::f32(&g, 16 * cols).map_err(e)?;
        let xv: Vec<u8> = (0..16 * cols).flat_map(|i| (((i * 7919) % 1000) as f32 / 500.0 - 1.0).to_le_bytes()).collect();
        x.write(0, &xv).map_err(e)?;
        let y = DevBuf::f32(&g, 16 * rows).map_err(e)?;
        // SAFETY: arithmetic only.
        let xq = DevBuf::new(&g, unsafe { (s.xq_bytes)(cols as i64, 16) }).map_err(e)?;
        let mut line = format!("{name:<26} {:>5} {:>14}", format!("{:?}", t.ty), format!("{rows} x {cols}"));
        let mut one = Vec::new();
        let mut check = String::new();
        for n in [1usize, 2, 4, 8, 16] {
            // SAFETY: x [n, cols] -> its split form; the product into y [n, rows].
            ffi::check(unsafe { (s.xq)(g.raw(), x.fp(), std::ptr::null(), 0, xq.ptr(), cols as i64, n as i64) }, "xq").map_err(e)?;
            let run = || -> Result<(), String> {
                // SAFETY: the repacked matrix (Q8_0: quants, scales at `at`), x's form, y.
                ffi::check(unsafe {
                    if kq {
                        (s.kr_mmvq)(g.raw(), ty, w.ptr(), xq.ptr(), y.fp(), cols as i64, rows as i64, n as i64)
                    } else {
                        (s.mmvq)(g.raw(), w.ptr(), w.ptr().cast::<u8>().add(at).cast(), xq.ptr(), y.fp(), cols as i64, rows as i64, n as i64)
                    }
                }, "mmvq").map_err(e)
            };
            run()?;
            g.sync().map_err(e)?;
            if kq && (n == 1 || n == 16) {
                let mut yb = vec![0u8; n * rows * 4];
                y.read(0, &mut yb).map_err(e)?;
                let yv = floats(&yb);
                if n == 1 {
                    one = yv[..rows].to_vec();
                } else {
                    let same = yv[..rows].iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits());
                    // the host's: the first rows dequantized, the columns' quants and scales read back
                    let rr = rows.min(256);
                    let wf = DevBuf::f32(&g, rr * cols).map_err(e)?;
                    // SAFETY: src's first rr rows -> float32.
                    ffi::check(unsafe { (k.dequant)(g.raw(), ty, src.ptr(), rr * cols, wf.fp()) }, "dequant").map_err(e)?;
                    g.sync().map_err(e)?;
                    let mut wb2 = vec![0u8; rr * cols * 4];
                    wf.read(0, &mut wb2).map_err(e)?;
                    let wv = floats(&wb2);
                    let mut qb = vec![0u8; xq.len];
                    xq.read(0, &mut qb).map_err(e)?;
                    let nb = cols / 32;
                    let sc_at = (nb * 512).div_ceil(256) * 256;
                    let xs = floats(&qb[sc_at..sc_at + 16 * nb * 4]);
                    let mut worst = 0f64;
                    for c in 0..16 {
                        let xd: Vec<f32> = (0..cols).map(|i| {
                            let q = qb[(i / 32) * 512 + ((i % 32) / 4) * 64 + c * 4 + (i % 4)] as i8;
                            q as f32 * xs[(i / 32) * 16 + c]
                        }).collect();
                        let (mut num, mut den) = (0f64, 0f64);
                        for r in 0..rr {
                            let rf: f64 = (0..cols).map(|i| wv[r * cols + i] as f64 * xd[i] as f64).sum();
                            let d = yv[c * rows + r] as f64 - rf;
                            num += d * d;
                            den += rf * rf;
                        }
                        worst = worst.max((num / den.max(1e-30)).sqrt());
                    }
                    // its expansion to half (a prompt pass's) against the same rows
                    let hb = DevBuf::new(&g, rr * cols * 2).map_err(e)?;
                    // SAFETY: w's first rr rows -> rr x cols halfs.
                    ffi::check(unsafe { (s.kr_f16)(g.raw(), ty, w.ptr(), rr as i64, cols as i64, hb.ptr().cast()) }, "kr f16").map_err(e)?;
                    g.sync().map_err(e)?;
                    let mut hh = vec![0u8; rr * cols * 2];
                    hb.read(0, &mut hh).map_err(e)?;
                    let mut hw = 0f32;
                    for (i, c) in hh.chunks_exact(2).enumerate() {
                        let h = f16(u16::from_le_bytes([c[0], c[1]]));
                        let r = wv[i];
                        hw = hw.max((h - r).abs() / r.abs().max(1e-3));
                    }
                    check = format!("  rel err {worst:.1e}, 16-column first == 1-column: {same}, half expansion {hw:.1e}");
                }
            }
            let reps = 20;
            let t0 = Instant::now();
            for _ in 0..reps {
                run()?;
            }
            g.sync().map_err(e)?;
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            line += &format!(" {:>9.0}", wb as f64 / dt / 1e9);
        }
        println!("{line}  GB/s{check}");
    }
    Ok(())
}
