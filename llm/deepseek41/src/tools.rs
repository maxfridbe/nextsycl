//! DeepSeek-V4.1's tools behind the command line's `info` and `kernels`.

use std::fmt::Write as _;
use std::path::Path;

use nextsycl_gguf::{GType, Gguf};

use crate::model::{Group, Model, Role};

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// `nextsycl info`: the geometry, every tensor checked by role, bytes by group
pub fn info(f: &Gguf) -> Result<String, String> {
    let mut out = String::new();
    let _ = writeln!(out, "arch     : deepseek41 ({}; {})", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"),
                     f.arch_meta("recipe").and_then(|v| v.as_str()).unwrap_or("?"));
    let m = Model::open(f).map_err(|e| e.0)?;
    let g = &m.g;
    let by = |r: u64| (0..g.n_layer).filter(|l| g.ratio(*l) == r).count();
    let _ = writeln!(out, "layers   : {} - window only {}, compressed 2:1 {}, 1:1 {}; kv sources {:?}, index sources {:?}, candidates from {:?} ({} blocks of {})",
                     g.n_layer, by(0), by(2), by(1), g.kv_sources, g.index_sources, g.cand_source, g.cand_top, g.cand_block);
    let _ = writeln!(out, "width    : hidden {}, vocab {}, context {}", g.n_embd, g.n_vocab, g.ctx);
    let _ = writeln!(out, "attention: {} heads of {} (rope {}), q LoRA {}, window {}, output {} groups of LoRA {}; indexer {} heads of {}, top {}",
                     g.n_head, g.head_dim, g.rope_dim, g.q_lora, g.window, g.o_groups, g.o_lora, g.idx_heads, g.idx_dim, g.idx_top_k);
    let _ = writeln!(out, "experts  : {} of {} per token + 1 shared, FFN {}, scale {}, SwiGLU limit {}", g.n_expert_used, g.n_expert, g.ffn_expert, g.expert_scale,
                     g.swiglu_limit);
    let _ = writeln!(out, "engram   : layers {:?}, {} rows of {} a token", g.engram_layers, g.engram_cols(), g.engram_dim);
    let _ = writeln!(out, "draft    : {} towers, {} experts top-{}, block {}, targets {:?}", g.n_mtp, g.mtp_expert, g.mtp_expert_used, g.mtp_block, g.mtp_targets);
    let errs = m.check();
    if errs.is_empty() {
        let _ = writeln!(out, "check    : every tensor present with the shape the runtime needs, none left over");
    } else {
        let _ = writeln!(out, "check    : {} problem(s)", errs.len());
        for e in errs.iter().take(40) {
            let _ = writeln!(out, "  {e}");
        }
    }
    let _ = writeln!(out, "bytes    :");
    let mut groups: std::collections::BTreeMap<Group, u64> = std::collections::BTreeMap::new();
    for ((grp, ty), (b, n)) in &m.bytes() {
        *groups.entry(*grp).or_default() += b;
        let _ = writeln!(out, "  {:<10} {:<10} {:>8.2} GiB  {:>4} tensors", format!("{grp:?}"), ty.name(), gib(*b), n);
    }
    for (grp, b) in &groups {
        let _ = writeln!(out, "  {:<21} {:>8.2} GiB", format!("{grp:?}"), gib(*b));
    }
    if !errs.is_empty() {
        print!("{out}");
        return Err(format!("{} layout problem(s)", errs.len()));
    }
    Ok(out)
}

/// E4M3FN byte -> float
fn e4m3(b: u8) -> f32 {
    let a = b & 0x7f;
    let (ex, mn) = ((a >> 3) as i32, (a & 7) as f32);
    let v = if ex == 0 { mn * 2f32.powi(-9) } else { (1.0 + mn / 8.0) * 2f32.powi(ex - 7) };
    if b & 0x80 != 0 { -v } else { v }
}
fn f16(h: u16) -> f32 {
    let (s, ex, m) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let bits = match (ex, m) {
        (0, 0) => s << 31,
        (0, _) => return if s == 1 { -(m as f32) * 2f32.powi(-24) } else { m as f32 * 2f32.powi(-24) },
        (31, _) => (s << 31) | (0xff << 23) | (m << 13),
        _ => (s << 31) | ((ex + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// A DQV3 payload decoded on the host (ds4-dfm-rs's vq.rs, the CPU oracle): [rows][cols]
pub fn vq_host(pay: &[u8], cb: &[u8], rows: usize, cols: usize, nbit: usize) -> Vec<f32> {
    let nidx = cols / 8;
    let mrow = nidx * 12 / 8;
    let gains = &pay[32..32 + rows * 2];
    let stream = &pay[32 + rows * 2..];
    let plane = &stream[rows * mrow..];
    let prow = nidx / 8;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let gn = f16(u16::from_le_bytes([gains[2 * r], gains[2 * r + 1]]));
        let s = &stream[r * mrow..];
        for i in 0..nidx {
            let bit = i * 12;
            let w = s[bit >> 3] as u32 | (s[(bit >> 3) + 1] as u32) << 8 | (*s.get((bit >> 3) + 2).unwrap_or(&0) as u32) << 16;
            let mut idx = (w >> (bit & 7)) & 0xFFF;
            if nbit > 12 {
                idx |= ((plane[r * prow + (i >> 3)] >> (i & 7)) as u32 & 1) << 12;
            }
            for d in 0..8 {
                out[r * cols + i * 8 + d] = e4m3(cb[idx as usize * 8 + d]) * gn;
            }
        }
    }
    out
}

fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dd, mut bb, mut mx) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dd += (x - y) * (x - y);
        bb += y * y;
        mx = mx.max((x - y).abs());
    }
    ((dd / bb.max(1e-30)).sqrt(), mx)
}

/// `nextsycl kernels`: the VQ kernels on real experts (13- and 12-bit layers, a tower) against the host decode, the
/// decode product against expanded weights x x on the host; the fp8_32x32 kernels likewise
pub fn kernels(model: &Path, gpu: usize) -> Result<(), String> {
    use crate::ops::Ops;
    use nextsycl_core::DevBuf;
    let e = |x: nextsycl_core::Error| x.0;
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let m = Model::open(&f).map_err(|e| e.0)?;
    let g = nextsycl_core::Gpu::open(gpu).map_err(e)?;
    let o = Ops::new(g.clone()).map_err(e)?;
    println!("gpu: {}", g.name);
    let (d, ff) = (m.g.n_embd as usize, m.g.ffn_expert as usize);
    let mut seed = 0x1234_5678u64;
    let mut rnd = || {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        ((seed.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32 - 0.5
    };
    for l in [0u64, 14, m.g.n_layer] {
        let t = m.tensor(l, Role::Experts).ok_or("no blob")?;
        let mut head = vec![0u8; 16 + 24];
        f.read_into(t, 0, &mut head).map_err(|e| e.0)?;
        let slot = |w: usize| u64::from_le_bytes(head[16 + w * 8..24 + w * 8].try_into().unwrap());
        let mut ph = [0u8; 32];
        f.read_into(t, slot(0), &mut ph).map_err(|e| e.0)?;
        let nc = u16::from_le_bytes([ph[6], ph[7]]) as usize;
        let nbit = if nc == 8192 { 13 } else { 12 };
        let cb_off = u64::from_le_bytes(ph[24..32].try_into().unwrap());
        let mut cb = vec![0u8; nc * 8];
        f.read_into(t, cb_off, &mut cb).map_err(|e| e.0)?;
        let cbd = DevBuf::new(&g, cb.len()).map_err(e)?;
        cbd.write(0, &cb).map_err(e)?;
        for (which, rows, cols) in [(0usize, ff, d), (2, d, ff)] {
            let len = (32 + rows * 2 + rows * cols * 12 / 64 + if nbit > 12 { rows * cols / 64 } else { 0 } + 4) as u64;
            let mut pay = vec![0u8; len as usize];
            f.read_into(t, slot(which), &mut pay).map_err(|e| e.0)?;
            let want = vq_host(&pay, &cb, rows, cols, nbit);
            let pd = DevBuf::new(&g, pay.len().next_multiple_of(256)).map_err(e)?;
            pd.write(0, &pay).map_err(e)?;
            let wd = DevBuf::f32(&g, rows * cols).map_err(e)?;
            o.vq_dequant(&pd, 0, pay.len(), &cbd, rows, cols, nbit, &wd, false).map_err(e)?;
            let got = wd.to_f32().map_err(e)?;
            let (rel, mx) = compare(&got, &want);
            // the decode product: 3 columns of x, against the host's W x
            let x: Vec<f32> = (0..3 * cols).map(|_| rnd()).collect();
            let xd = DevBuf::from_f32(&g, &x).map_err(e)?;
            let mut tb: Vec<u8> = (pd.ptr() as u64).to_le_bytes().to_vec();
            for v in [3i32, 0, 0, 1, 2, 0, 0, 0, 0, 0] {
                tb.extend(v.to_le_bytes());
            }
            let tab = DevBuf::new(&g, tb.len()).map_err(e)?;
            tab.write(0, &tb).map_err(e)?;
            let yd = DevBuf::f32(&g, 3 * rows).map_err(e)?;
            let t0 = std::time::Instant::now();
            o.vq_mmv(&tab, 1, (&cbd, None), &xd, (&yd, 0), 3, rows, cols, nbit, 0).map_err(e)?;
            g.sync().map_err(e)?;
            let us = t0.elapsed().as_secs_f64() * 1e6;
            let y = yd.to_f32().map_err(e)?;
            let mut yw = vec![0f32; 3 * rows];
            for c in 0..3 {
                for r in 0..rows {
                    yw[c * rows + r] = (0..cols).map(|k| want[r * cols + k] as f64 * x[c * cols + k] as f64).sum::<f64>() as f32;
                }
            }
            let (yrel, _) = compare(&y, &yw);
            println!("{:<26} {}-bit [{rows} x {cols}]: expanded rel err {rel:.2e} (max {mx:.2e}), 3-column product rel err {yrel:.2e} ({us:.0} us)",
                     format!("{} {}", t.name, ["gate", "up", "down"][which]), nbit);
        }
    }
    // a decode layer's products: 6 experts' gate | up in one launch, their downs in another, one token - a 12-bit layer
    // (20) and a 13-bit one (0); 8 sets of 6 experts in turn (the indices from VRAM, not L2, as in a pass), each variant
    // of the kernel (NS_DS41_VQ_BENCH, default "0,1") against the first, after half a second of warm-up (the clocks up)
    for l in [20u64, 0] {
        let t = m.tensor(l, Role::Experts).ok_or("no blob")?;
        let (n_e, sets) = (6usize, 8usize);
        let raw = {
            let mut v = vec![0u8; (16 + 384 * 24) as usize];
            f.read_into(t, 0, &mut v).map_err(|e| e.0)?;
            v
        };
        let slot = |x: usize, w: usize| u64::from_le_bytes(raw[16 + (x * 3 + w) * 8..24 + (x * 3 + w) * 8].try_into().unwrap());
        let mut ph = [0u8; 32];
        f.read_into(t, slot(0, 0), &mut ph).map_err(|e| e.0)?;
        let ncb = u16::from_le_bytes([ph[6], ph[7]]) as usize;
        let ncb = if ncb == 0 { 65536 } else { ncb };
        let nbit = if ncb > 4096 { 13 } else { 12 };
        let cb_off = u64::from_le_bytes(ph[24..32].try_into().unwrap());
        let mut cb = vec![0u8; ncb * 8];
        f.read_into(t, cb_off, &mut cb).map_err(|e| e.0)?;
        let cbd = DevBuf::new(&g, cb.len()).map_err(e)?;
        cbd.write(0, &cb).map_err(e)?;
        let span = (slot(1, 0) - slot(0, 0)) as usize;
        let total = n_e * sets;
        let buf = DevBuf::new(&g, total * span.next_multiple_of(256)).map_err(e)?;
        let (mut gts, mut dts) = (Vec::new(), Vec::new());
        for s in 0..sets {
            let mut tb: Vec<u8> = Vec::new();
            let mut meta: Vec<i32> = Vec::new();
            let mut db: Vec<u8> = Vec::new();
            let mut dmeta: Vec<i32> = Vec::new();
            for x in 0..n_e {
                let ex = s * n_e + x;
                let src = ex * 7 + 3;
                let s0 = slot(src, 0);
                let mut v = vec![0u8; span];
                f.read_into(t, s0, &mut v).map_err(|e| e.0)?;
                let at = ex * span.next_multiple_of(256);
                buf.write(at, &v).map_err(e)?;
                let base = buf.ptr() as u64 + at as u64;
                for w in 0..2 {
                    tb.extend((base + slot(src, w) - s0).to_le_bytes());
                    meta.extend([1, (w * n_e + x) as i32, 0, 0, 0, 0, 0, 0, 0, 0]);
                }
                db.extend((base + slot(src, 2) - s0).to_le_bytes());
                dmeta.extend([1, x as i32, x as i32, 0, 0, 0, 0, 0, 0, 0]);
            }
            tb.extend(meta.iter().flat_map(|v| v.to_le_bytes()));
            db.extend(dmeta.iter().flat_map(|v| v.to_le_bytes()));
            let gt = DevBuf::new(&g, tb.len()).map_err(e)?;
            gt.write(0, &tb).map_err(e)?;
            let dt = DevBuf::new(&g, db.len()).map_err(e)?;
            dt.write(0, &db).map_err(e)?;
            gts.push(gt);
            dts.push(dt);
        }
        let x: Vec<f32> = (0..d).map(|_| rnd()).collect();
        let xd = DevBuf::from_f32(&g, &x).map_err(e)?;
        let gu = DevBuf::f32(&g, 2 * n_e * ff).map_err(e)?;
        let h: Vec<f32> = (0..n_e * ff).map(|_| rnd()).collect();
        let hd = DevBuf::from_f32(&g, &h).map_err(e)?;
        let dn = DevBuf::f32(&g, n_e * d).map_err(e)?;
        let cbf = DevBuf::f32(&g, cb.len()).map_err(e)?;
        o.vq_codebook_f32(&cbd, cb.len(), &cbf).map_err(e)?;
        let cbh = DevBuf::new(&g, cb.len() * 2).map_err(e)?;
        o.vq_codebook_f16(&cbd, cb.len(), &cbh).map_err(e)?;
        let variants: Vec<i32> = std::env::var("NS_DS41_VQ_BENCH").unwrap_or_else(|_| "0,1".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
        for (name, tabs, n, x, y, rows, cols) in [("gate | up", &gts, 2 * n_e, &xd, &gu, ff, d), ("down", &dts, n_e, &hd, &dn, d, ff)] {
            let mut first: Option<Vec<f32>> = None;
            for &v in &variants {
                let t0 = std::time::Instant::now();
                while t0.elapsed().as_secs_f64() < 0.5 {
                    for tab in tabs.iter() {
                        o.vq_mmv(tab, n, (&cbd, Some(if v >= 2 && v != 4 { &cbh } else { &cbf })), x, (y, 0), n, rows, cols, nbit, v).map_err(e)?;
                    }
                    g.sync().map_err(e)?;
                }
                let it = 40;
                let t0 = std::time::Instant::now();
                for i in 0..it {
                    o.vq_mmv(&tabs[i % sets], n, (&cbd, Some(if v >= 2 && v != 4 { &cbh } else { &cbf })), x, (y, 0), n, rows, cols, nbit, v).map_err(e)?;
                }
                g.sync().map_err(e)?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / it as f64;
                o.vq_mmv(&tabs[0], n, (&cbd, Some(if v >= 2 && v != 4 { &cbh } else { &cbf })), x, (y, 0), n, rows, cols, nbit, v).map_err(e)?;
                let got = y.view(0, n * rows * 4).map_err(e)?.to_f32().map_err(e)?;
                let rel = match &first {
                    Some(f0) => compare(&got, f0).0,
                    None => {
                        first = Some(got);
                        0.0
                    }
                };
                let bytes = n * rows * (cols * nbit / 64 + 2);
                println!("vq_mmv v{v} {nbit}-bit {name:<10} {n} matrices [{rows} x {cols}], 1 token: {us:.1} us ({:.0} GB/s of indices), vs v{}: {rel:.1e}",
                         bytes as f64 / us / 1e3, variants[0]);
            }
        }
    }
    // the memory's own rate: 256 MiB copied by a kernel (a read and a write a byte)
    {
        let n = 256usize << 20;
        let (a, b) = (DevBuf::new(&g, n).map_err(e)?, DevBuf::new(&g, n).map_err(e)?);
        o.kcopy(&b, 0, &a, 0, n).map_err(e)?;
        g.sync().map_err(e)?;
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            o.kcopy(&b, 0, &a, 0, n).map_err(e)?;
        }
        g.sync().map_err(e)?;
        let s = t0.elapsed().as_secs_f64() / 10.0;
        println!("copy kernel: 256 MiB in {:.0} us ({:.0} GB/s read + write)", s * 1e6, 2.0 * n as f64 / s / 1e9);
    }
    // q4_K decode products: q_b and o_b of layer 20, against the shared kernels' expansion
    for role in [Role::QB, Role::OutB] {
        let t = m.tensor(20, role).ok_or("no tensor")?;
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        let raw = f.read(t).map_err(|e| e.0)?;
        let wd = DevBuf::new(&g, raw.len()).map_err(e)?;
        wd.write(0, &raw).map_err(e)?;
        let wf = DevBuf::f32(&g, rows * cols).map_err(e)?;
        o.dequant(12, &wd, 0, raw.len(), rows * cols, &wf).map_err(e)?;
        let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();
        let xd = DevBuf::from_f32(&g, &x).map_err(e)?;
        let yd = DevBuf::f32(&g, rows).map_err(e)?;
        let yw = DevBuf::f32(&g, rows).map_err(e)?;
        o.gemm_at(1, rows, cols, (&xd, 0, cols), (&wf, 0), (&yw, 0, rows), false).map_err(e)?;
        for _ in 0..3 {
            o.q4k_mv(&wd, rows, cols, (&xd, 0, cols), None, (&yd, 0, rows), 1).map_err(e)?;
        }
        g.sync().map_err(e)?;
        let it = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..it {
            o.q4k_mv(&wd, rows, cols, (&xd, 0, cols), None, (&yd, 0, rows), 1).map_err(e)?;
        }
        g.sync().map_err(e)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / it as f64;
        let (rel, _) = compare(&yd.to_f32().map_err(e)?, &yw.to_f32().map_err(e)?);
        println!("q4k_mv {:<24} [{rows} x {cols}]: rel err {rel:.2e}, {us:.1} us ({:.0} GB/s)", t.name, raw.len() as f64 / us / 1e3);
        // the shared kernels (Strata's): Q8_1 of x, then the product
        let q8 = DevBuf::new(&g, o.q8_1_bytes(cols, 1)).map_err(e)?;
        o.quantize_q8_1((&xd, 0), &q8, cols, 1).map_err(e)?;
        for _ in 0..3 {
            o.mmvq(12, (&wd, 0), raw.len(), &q8, (&yd, 0), cols, rows, 1).map_err(e)?;
        }
        g.sync().map_err(e)?;
        let t0 = std::time::Instant::now();
        for _ in 0..it {
            o.mmvq(12, (&wd, 0), raw.len(), &q8, (&yd, 0), cols, rows, 1).map_err(e)?;
        }
        g.sync().map_err(e)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / it as f64;
        println!("shared mmvq {:<19} [{rows} x {cols}]: {us:.1} us ({:.0} GB/s)", t.name, raw.len() as f64 / us / 1e3);
    }
    // an fp8_32x32 matrix: the Engram's wkv, rows [0, 64)
    if let Some(t) = m.tensor(m.g.engram_layers.first().copied().unwrap_or(1), Role::EngramWkv).filter(|t| t.ty == GType::Fp8B32) {
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        let raw = f.read(t).map_err(|e| e.0)?;
        let wd = DevBuf::new(&g, raw.len()).map_err(e)?;
        wd.write(0, &raw).map_err(e)?;
        let n = 64;
        let ex = DevBuf::f32(&g, n * cols).map_err(e)?;
        o.fp8_dequant(&wd, 0, rows, cols, 0, n, &ex, false).map_err(e)?;
        let got = ex.to_f32().map_err(e)?;
        let tc = cols / 32;
        let sc = &raw[rows * cols..];
        let mut want = vec![0f32; n * cols];
        for r in 0..n {
            for c in 0..cols {
                let s = sc[(r / 32) * tc + c / 32];
                want[r * cols + c] = e4m3(raw[r * cols + c]) * f32::from_bits(if s == 0 { 0x0040_0000 } else { (s as u32) << 23 });
            }
        }
        let (rel, _) = compare(&got, &want);
        let x: Vec<f32> = (0..2 * cols).map(|_| rnd()).collect();
        let xd = DevBuf::from_f32(&g, &x).map_err(e)?;
        let yd = DevBuf::f32(&g, 2 * rows).map_err(e)?;
        o.fp8_mmv(&wd, rows, cols, (&xd, 0), (&yd, 0), 2).map_err(e)?;
        let y = yd.to_f32().map_err(e)?;
        let mut yrel = 0f64;
        for c in 0..2 {
            let yw: Vec<f32> = (0..n).map(|r| (0..cols).map(|k| want[r * cols + k] as f64 * x[c * cols + k] as f64).sum::<f64>() as f32).collect();
            yrel = yrel.max(compare(&y[c * rows..c * rows + n], &yw).0);
        }
        println!("{:<26} [{rows} x {cols}]: expanded rel err {rel:.2e}, 2-column product rel err {yrel:.2e} (first {n} rows)", t.name);
    }
    Ok(())
}
