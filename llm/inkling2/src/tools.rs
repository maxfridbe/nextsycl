//! `nextsycl llm kernels <model.gguf> [--gpu N]`: this engine's weight kernels on the model's own matrices - each
//! type's expansion against the GGUF block format decoded here on the host (bit for bit; with NS_INK2_DEQ_REF=DIR also
//! against ggml's dequantize_row_* output, `reference/inkling2/deq-ref.cpp`), the decode products against the host's
//! float64 dot of the decoded rows, a 16-column pass's first column against the one-column pass's (bit for bit), and
//! each product's speed as GB/s of weights.

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

/// NVFP4's UE4M3 block scale, halved (its codes are twice their E2M1 values)
fn ue4m3(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let (e, m) = (((x >> 3) & 0xF) as i32, (x & 7) as f32);
    if e == 0 { m * 2f32.powi(-10) } else { (8.0 + m) * 2f32.powi(e - 11) }
}

const E2M1X2: [f32; 16] = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0];

/// A row's blocks decoded on the host, from the formats
pub fn decode(ty: GType, raw: &[u8], n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    match ty {
        GType::F32 => v.extend(raw[..n * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
        GType::BF16 => v.extend(raw[..n * 2].chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))),
        GType::Q8_0 => {
            for b in raw[..n / 32 * 34].chunks_exact(34) {
                let d = f16(u16::from_le_bytes([b[0], b[1]]));
                v.extend(b[2..].iter().map(|q| d * (*q as i8) as f32));
            }
        }
        GType::NVFP4 => {
            for b in raw[..n / 64 * 36].chunks_exact(36) {
                for s in 0..4 {
                    let d = ue4m3(b[s]);
                    let mut sub = [0f32; 16];
                    for j in 0..8 {
                        let c = b[4 + s * 8 + j];
                        sub[j] = E2M1X2[(c & 15) as usize] * d;
                        sub[j + 8] = E2M1X2[(c >> 4) as usize] * d;
                    }
                    v.extend(sub);
                }
            }
        }
        _ => {}
    }
    v
}

pub fn kernels(model: &Path, gpu: usize) -> Result<(), String> {
    let e = |x: nextsycl_core::Error| x.0;
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let g = nextsycl_core::Gpu::open(gpu).map_err(e)?;
    let k = ffi::api().map_err(e)?;
    let refdir = std::env::var("NS_INK2_DEQ_REF").ok();
    println!("gpu: {}{}", g.name, refdir.as_ref().map_or(String::new(), |d| format!("; ggml's rows from {d}")));
    // one matrix of each type the file has (an expert stack's first expert), the head
    let mut names: Vec<String> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for t in &f.tensors {
        if t.shape.len() >= 2 && seen.insert((t.ty.code(), t.name.contains("exps"))) && t.ty != GType::F32 {
            names.push(t.name.clone());
        }
    }
    for n in ["blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.attn_q.weight", "output.weight"] {
        if !names.iter().any(|x| x == n) && f.tensor(n).is_some() {
            names.push(n.to_string());
        }
    }
    println!("{:<34} {:>6} {:>12} {:>10} {:>10} {:>12} {:>9} {:>9}", "matrix", "type", "rows x cols", "expand", "vs ggml", "mv rel err", "1 GB/s", "16 GB/s");
    let mut bad = 0;
    for name in &names {
        let t = f.tensor(name).ok_or("a tensor")?;
        let cols = t.shape[t.shape.len() - 1] as usize;
        let rows_all = t.shape[t.shape.len() - 2] as usize;
        let ty = t.ty;
        // SAFETY: arithmetic only.
        if unsafe { (k.type_supported)(ty.code() as c_int, cols as i64) } == 0 {
            println!("{name:<34} {:>6} not a type this engine takes", ty.name());
            continue;
        }
        let row_bytes = ty.bytes(cols as u64).ok_or("row size")? as usize;
        let rows = rows_all.min(4096);
        let mut raw = vec![0u8; rows * row_bytes];
        f.read_into(t, 0, &mut raw).map_err(|x| x.0)?;
        let src = DevBuf::new(&g, raw.len()).map_err(e)?;
        src.write(0, &raw).map_err(e)?;
        // the expansion against the host's decode, bit for bit
        let out = DevBuf::f32(&g, rows * cols).map_err(e)?;
        // SAFETY: rows x cols values of the type.
        ffi::check(unsafe { (k.dequant)(g.raw(), ty.code() as c_int, src.ptr(), (rows * cols) as i64, out.fp()) }, "dequant").map_err(e)?;
        let mine = out.to_f32().map_err(e)?;
        let host: Vec<f32> = (0..rows).flat_map(|r| decode(ty, &raw[r * row_bytes..(r + 1) * row_bytes], cols)).collect();
        let diff = mine.iter().zip(&host).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let expand = if diff == 0 { "exact".to_string() } else { format!("{diff} differ") };
        bad += diff;
        // ggml's own rows (deq-ref.cpp's dump: <name>.f32, the first rows)
        let vs = match &refdir {
            Some(d) => match std::fs::read(Path::new(d).join(format!("{name}.f32"))) {
                Ok(b) => {
                    let theirs: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let n = theirs.len().min(mine.len());
                    let d = mine[..n].iter().zip(&theirs[..n]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    bad += d;
                    if d == 0 { format!("exact {n}") } else { format!("{d} differ") }
                }
                Err(_) => "-".into(),
            },
            None => "-".into(),
        };
        // the products: 16 columns of a fixed pseudo-random input
        let tt = 16usize;
        let mut st = 0x2545F4914F6CDD1Du64;
        let x: Vec<f32> = (0..tt * cols).map(|_| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }).collect();
        let xd = DevBuf::from_f32(&g, &x).map_err(e)?;
        let y16 = DevBuf::f32(&g, tt * rows).map_err(e)?;
        let y1 = DevBuf::f32(&g, rows).map_err(e)?;
        // SAFETY: the matrix's rows, x [16, cols], y [16, rows]; then the first column alone.
        unsafe {
            ffi::check((k.mv)(g.raw(), ty.code() as c_int, src.ptr(), xd.fp(), y16.fp(), cols as i64, rows as i64, tt as i64), "mv").map_err(e)?;
            ffi::check((k.mv)(g.raw(), ty.code() as c_int, src.ptr(), xd.fp(), y1.fp(), cols as i64, rows as i64, 1), "mv").map_err(e)?;
        }
        let a16 = y16.to_f32().map_err(e)?;
        let a1 = y1.to_f32().map_err(e)?;
        let same = a1.iter().zip(&a16[..rows]).all(|(a, b)| a.to_bits() == b.to_bits());
        if !same {
            bad += 1;
        }
        let mut worst = 0f64;
        for c in 0..tt {
            for r in 0..rows.min(256) {
                let want: f64 = host[r * cols..(r + 1) * cols].iter().zip(&x[c * cols..(c + 1) * cols]).map(|(a, b)| *a as f64 * *b as f64).sum();
                let scale: f64 = host[r * cols..(r + 1) * cols].iter().zip(&x[c * cols..(c + 1) * cols]).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
                worst = worst.max((a16[c * rows + r] as f64 - want).abs() / scale.max(1e-30));
            }
        }
        let speed = |cols_n: usize| -> Result<f64, String> {
            let reps = 20;
            g.sync().map_err(e)?;
            let t0 = Instant::now();
            for _ in 0..reps {
                // SAFETY: as above.
                ffi::check(unsafe { (k.mv)(g.raw(), ty.code() as c_int, src.ptr(), xd.fp(), y16.fp(), cols as i64, rows as i64, cols_n as i64) }, "mv").map_err(e)?;
            }
            g.sync().map_err(e)?;
            Ok(raw.len() as f64 * reps as f64 / t0.elapsed().as_secs_f64() / 1e9)
        };
        let (s1, s16) = (speed(1)?, speed(16)?);
        println!("{name:<34} {:>6} {:>12} {expand:>10} {vs:>10} {worst:>12.2e} {s1:>9.0} {s16:>9.0}{}", ty.name(), format!("{rows}x{cols}"),
                 if same { "" } else { "  (16-column pass's first column differs from the one-column pass)" });
    }
    if bad > 0 {
        return Err(format!("{bad} values differ"));
    }
    Ok(())
}
