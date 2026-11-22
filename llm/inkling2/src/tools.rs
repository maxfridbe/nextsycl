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
    let silo = ffi::silo().map_err(e)?;
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
    for n in ["blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.attn_q.weight", "blk.0.attn_output.weight", "blk.0.attn_rel.weight", "output.weight"] {
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
        // the silo's products on the matrix repacked: against the same host dot (the activations quantized as Q8_0: a
        // larger error), a 16-row pass's first row against the one-row pass's, and the speed
        let Some(s) = silo else { continue };
        let code = ty.code() as c_int;
        // SAFETY: arithmetic only.
        let fl = unsafe { (s.fmv_supported)(code, cols as i64) } != 0;
        if unsafe { (s.supported)(code, cols as i64) } == 0 && !fl {
            continue;
        }
        let (w, xq) = if fl {
            (src.view(0, src.len).map_err(e)?, None)
        } else {
            // SAFETY: arithmetic only; then the repacking and the activations' quantization.
            let p = DevBuf::new(&g, unsafe { (s.packed_bytes)(code, rows as i64, cols as i64) }).map_err(e)?;
            let q = DevBuf::new(&g, unsafe { (s.xq_bytes)(tt as i64, cols as i64) }).map_err(e)?;
            let q1 = DevBuf::new(&g, unsafe { (s.xq_bytes)(1, cols as i64) }).map_err(e)?;
            unsafe {
                ffi::check((s.pack)(g.raw(), code, src.ptr(), rows as i64, cols as i64, p.ptr()), "pack").map_err(e)?;
                ffi::check((s.xq)(g.raw(), xd.fp(), q.ptr(), tt as i64, cols as i64), "xq").map_err(e)?;
                ffi::check((s.xq)(g.raw(), xd.fp(), q1.ptr(), 1, cols as i64), "xq").map_err(e)?;
            }
            // the repacked rows expanded to half against the host's decode rounded to half
            let h16 = DevBuf::new(&g, rows * cols * 2).map_err(e)?;
            // SAFETY: all rows of the repacked matrix -> half.
            ffi::check(unsafe { (s.unpack_f16)(g.raw(), code, p.ptr(), rows as i64, 0, rows as i64, cols as i64, h16.ptr().cast()) }, "unpack").map_err(e)?;
            let mut hb = vec![0u8; rows * cols * 2];
            h16.read(0, &mut hb).map_err(e)?;
            // half's rounding: within 2^-11 of the value (3e-8 near zero)
            let hd = hb.chunks_exact(2).zip(&host).filter(|(c, v)| (f16(u16::from_le_bytes([c[0], c[1]])) - **v).abs() > v.abs() / 2048.0 + 3e-8).count();
            if hd > 0 {
                println!("{name:<34} silo unpack: {hd} values further than half's rounding from the host's decode");
                bad += hd;
            }
            (p, Some((q, q1)))
        };
        let run = |n: usize, y: &DevBuf| -> Result<(), String> {
            // SAFETY: the matrix (repacked, or float), the activations, y [n, rows].
            ffi::check(unsafe {
                match &xq {
                    Some((q, q1)) => (s.mv)(g.raw(), code, w.ptr(), if n == 1 { q1.ptr() } else { q.ptr() }, y.fp(), cols as i64, rows as i64, n as i64),
                    None => (s.fmv)(g.raw(), code, w.ptr(), xd.fp(), y.fp(), cols as i64, rows as i64, n as i64),
                }
            }, "silo product").map_err(e)
        };
        run(tt, &y16)?;
        run(1, &y1)?;
        let (b16, b1) = (y16.to_f32().map_err(e)?, y1.to_f32().map_err(e)?);
        let same = b1.iter().zip(&b16[..rows]).all(|(a, b)| a.to_bits() == b.to_bits());
        if !same {
            bad += 1;
        }
        let mut worst = 0f64;
        for c in 0..tt {
            for r in 0..rows.min(256) {
                let want: f64 = host[r * cols..(r + 1) * cols].iter().zip(&x[c * cols..(c + 1) * cols]).map(|(a, b)| *a as f64 * *b as f64).sum();
                let scale: f64 = host[r * cols..(r + 1) * cols].iter().zip(&x[c * cols..(c + 1) * cols]).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
                worst = worst.max((b16[c * rows + r] as f64 - want).abs() / scale.max(1e-30));
            }
        }
        let speed = |n: usize| -> Result<f64, String> {
            let reps = 20;
            g.sync().map_err(e)?;
            let t0 = Instant::now();
            for _ in 0..reps {
                run(n, &y16)?;
            }
            g.sync().map_err(e)?;
            Ok(raw.len() as f64 * reps as f64 / t0.elapsed().as_secs_f64() / 1e9)
        };
        let (s1, s16) = (speed(1)?, speed(16)?);
        println!("{:<34} {:>6} {:>12} {:>10} {:>10} {worst:>12.2e} {s1:>9.0} {s16:>9.0}{}", "  silo", ty.name(), "", if xq.is_some() { "repacked" } else { "float" }, "",
                 if same { "" } else { "  (16-row pass's first row differs from the one-row pass)" });
    }
    if let Some(s) = silo {
        stream(&f, &g, s)?;
    }
    if bad > 0 {
        return Err(format!("{bad} values differ"));
    }
    Ok(())
}

/// The silo's decode products at the size they run, out of DRAM: the whole head (one row), and a layer's routed experts
/// (gate | up with SiTU, then the down products and their sum) for one row's 8 experts, drawn anew each call
fn stream(f: &Gguf, g: &std::sync::Arc<nextsycl_core::Gpu>, s: &ffi::Silo) -> Result<(), String> {
    let e = |x: nextsycl_core::Error| x.0;
    let up = |name: &str| -> Result<(DevBuf, GType, usize, usize), String> {
        let t = f.tensor(name).ok_or(format!("no {name}"))?;
        let cols = t.shape[t.shape.len() - 1] as usize;
        let rows = t.shape[..t.shape.len() - 1].iter().product::<u64>() as usize;
        let raw = f.read(t).map_err(|x| x.0)?;
        let src = DevBuf::new(g, raw.len()).map_err(e)?;
        src.write(0, &raw).map_err(e)?;
        let code = t.ty.code() as c_int;
        // SAFETY: arithmetic only; then the repacking.
        let p = DevBuf::new(g, unsafe { (s.packed_bytes)(code, rows as i64, cols as i64) }).map_err(e)?;
        ffi::check(unsafe { (s.pack)(g.raw(), code, src.ptr(), rows as i64, cols as i64, p.ptr()) }, "pack").map_err(e)?;
        g.sync().map_err(e)?;
        Ok((p, t.ty, rows, cols))
    };
    let reps = 50;
    let packs = |name: &str| {
        // SAFETY: arithmetic only.
        f.tensor(name).is_some_and(|t| unsafe { (s.supported)(t.ty.code() as c_int, t.shape[t.shape.len() - 1] as i64) } != 0)
    };
    if !packs("output.weight") || !packs("blk.0.ffn_gate_up_exps.weight") || !packs("blk.0.ffn_down_exps.weight") {
        println!("silo bandwidth: the head or the experts are not types it repacks (their float products are above)");
        return Ok(());
    }
    let (head, hty, hrows, hcols) = up("output.weight")?;
    let x = DevBuf::from_f32(g, &(0..hcols).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect::<Vec<f32>>()).map_err(e)?;
    // SAFETY: arithmetic only; x quantized.
    let xq = DevBuf::new(g, unsafe { (s.xq_bytes)(1, hcols as i64) }).map_err(e)?;
    ffi::check(unsafe { (s.xq)(g.raw(), x.fp(), xq.ptr(), 1, hcols as i64) }, "xq").map_err(e)?;
    let y = DevBuf::f32(g, hrows).map_err(e)?;
    g.sync().map_err(e)?;
    let t0 = Instant::now();
    for _ in 0..reps {
        // SAFETY: the repacked head, x's quants, y [rows].
        ffi::check(unsafe { (s.mv)(g.raw(), hty.code() as c_int, head.ptr(), xq.ptr(), y.fp(), hcols as i64, hrows as i64, 1) }, "mv").map_err(e)?;
    }
    g.sync().map_err(e)?;
    println!("silo head {}x{hcols} {}, one row: {:.0} GB/s", hrows, hty.name(), head.len as f64 * reps as f64 / t0.elapsed().as_secs_f64() / 1e9);
    drop(head);
    let (gu, gty, gr, gc) = up("blk.0.ffn_gate_up_exps.weight")?;
    let (dn, dty, dr, dc) = up("blk.0.ffn_down_exps.weight")?;
    let n_exp = f.arch_meta("expert_count").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let (k8, ff) = (8usize, gr / n_exp / 2);
    // 50 sets of 8 experts (a fixed pseudo-random draw), weights 1/8
    let mut st = 12345u64;
    let ids: Vec<i32> = (0..reps * k8).map(|_| {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 33) % n_exp as u64) as i32
    }).collect();
    let idb = DevBuf::new(g, ids.len() * 4).map_err(e)?;
    idb.write(0, &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(e)?;
    let wts = DevBuf::from_f32(g, &vec![0.125f32; k8]).map_err(e)?;
    // SAFETY: arithmetic only.
    let aq = DevBuf::new(g, unsafe { (s.xq_bytes)(k8 as i64, ff as i64) }).map_err(e)?;
    let xd = DevBuf::f32(g, dr / n_exp).map_err(e)?;
    let xq2 = DevBuf::new(g, unsafe { (s.xq_bytes)(1, gc as i64) }).map_err(e)?;
    ffi::check(unsafe { (s.xq)(g.raw(), x.fp(), xq2.ptr(), 1, gc as i64) }, "xq").map_err(e)?;
    g.sync().map_err(e)?;
    let t0 = Instant::now();
    for r in 0..reps {
        let ip: *const i32 = unsafe { idb.ptr().cast::<i32>().add(r * k8) };
        // SAFETY: the stacks, 8 ids, the row's quants, the act's, the output row.
        unsafe {
            ffi::check((s.moe_gu)(g.raw(), gty.code() as c_int, gu.ptr(), n_exp as i64, std::ptr::null(), ip, xq2.ptr(), aq.ptr(), 1, k8 as i64, gc as i64, ff as i64,
                                  4.0, 25.0), "moe gu").map_err(e)?;
            ffi::check((s.moe_down)(g.raw(), dty.code() as c_int, dn.ptr(), n_exp as i64, std::ptr::null(), ip, wts.fp(), aq.ptr(), xd.fp(), 1, k8 as i64, dc as i64,
                                    (dr / n_exp) as i64), "moe down").map_err(e)?;
        }
    }
    g.sync().map_err(e)?;
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let bytes = k8 * (gu.len / n_exp + dn.len / n_exp);
    println!("silo experts {} (8 of {n_exp}: gate|up {}x{gc}, down {}x{dc}): {:.1} us a row, {:.0} GB/s", gty.name(), gr / n_exp, dr / n_exp, dt * 1e6,
             bytes as f64 / dt / 1e9);
    Ok(())
}
