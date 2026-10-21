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

/// A float of magnitude below 2 (normal or zero, as the bench's) to fp16's bits, rounded to nearest even
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    if e <= 0 {
        return sign;
    }
    let m = b & 0x7F_FFFF;
    let (mut h, rest) = (((e as u32) << 10) | (m >> 13), m & 0x1FFF);
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
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

/// `nextsycl kernels` on a llama.cpp-form file: each expert type's expansion and decode product on real experts (one of
/// each type, the first and a middle expert). The product against the host's W x on the weights the expansion made - or
/// on a reference's when `NS_DS41_STD_DIR` holds `<tag>.ref.f32` (ggml's own dequantize_row_* on `<tag>.raw`, which this
/// writes beside `<tag>.gpu.f32`), and the expansion against it
fn kernels_std(f: &Gguf, m: &Model, g: &std::sync::Arc<nextsycl_core::Gpu>, o: &crate::ops::Ops) -> Result<(), String> {
    use nextsycl_core::DevBuf;
    let e = |x: nextsycl_core::Error| x.0;
    let gg = &m.g;
    let (d, ff) = (gg.n_embd as usize, gg.ffn_expert as usize);
    let dir = std::env::var("NS_DS41_STD_DIR").ok().map(std::path::PathBuf::from);
    let mut seed = 0x1234_5678u64;
    let mut rnd = || {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        ((seed.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32 - 0.5
    };
    let mut seen = std::collections::BTreeSet::new();
    for l in 0..gg.n_layer {
        for (which, role, rows, cols) in [(0usize, Role::ExpGate, ff, d), (1, Role::ExpUp, ff, d), (2, Role::ExpDown, d, ff)] {
            let t = m.tensor(l, role).ok_or("an expert tensor is missing")?;
            if !seen.insert((t.ty.code(), which == 2)) {
                continue;
            }
            let mat = t.bytes as usize / gg.n_expert as usize;
            for ex in [0u64, gg.n_expert / 2] {
                let mut raw = vec![0u8; mat];
                f.read_into(t, ex * mat as u64, &mut raw).map_err(|x| x.0)?;
                let src = DevBuf::new(g, mat.next_multiple_of(256)).map_err(e)?;
                src.write(0, &raw).map_err(e)?;
                let wd = DevBuf::f32(g, rows * cols).map_err(e)?;
                o.std_dequant(t.ty.code(), &src, 0, mat, rows, cols, &wd, false).map_err(e)?;
                let got = wd.to_f32().map_err(e)?;
                let tag = format!("{}_{}_L{l}_e{ex}", t.ty.name(), ["gate", "up", "down"][which]);
                let mut note = String::new();
                let mut want = got.clone();
                if let Some(dir) = &dir {
                    std::fs::write(dir.join(format!("{tag}.raw")), &raw).map_err(|x| x.to_string())?;
                    std::fs::write(dir.join(format!("{tag}.gpu.f32")), got.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|x| x.to_string())?;
                    if let Ok(b) = std::fs::read(dir.join(format!("{tag}.ref.f32"))) {
                        let r: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        if r.len() != got.len() {
                            return Err(format!("{tag}: the reference has {} values, {} expected", r.len(), got.len()));
                        }
                        let (rel, mx) = compare(&got, &r);
                        note = format!("expansion vs ggml: rel err {rel:.2e} (max {mx:.2e}); ");
                        want = r;
                    }
                }
                // the decode product: 3 columns of x, against the host's W x
                let x: Vec<f32> = (0..3 * cols).map(|_| rnd()).collect();
                let xd = DevBuf::from_f32(g, &x).map_err(e)?;
                let mut tb: Vec<u8> = (src.ptr() as u64).to_le_bytes().to_vec();
                for v in [3i32, 0, 0, 1, 2, 0, 0, 0, 0, 0] {
                    tb.extend(v.to_le_bytes());
                }
                let tab = DevBuf::new(g, tb.len()).map_err(e)?;
                tab.write(0, &tb).map_err(e)?;
                let yd = DevBuf::f32(g, 3 * rows).map_err(e)?;
                o.std_mmv(t.ty.code(), &tab, 1, &xd, (&yd, 0), 3, rows, cols).map_err(e)?;
                g.sync().map_err(e)?;
                let y = yd.to_f32().map_err(e)?;
                let mut yw = vec![0f32; 3 * rows];
                for c in 0..3 {
                    for r in 0..rows {
                        yw[c * rows + r] = (0..cols).map(|k| want[r * cols + k] as f64 * x[c * cols + k] as f64).sum::<f64>() as f32;
                    }
                }
                let (yrel, _) = compare(&y, &yw);
                // six experts' worth in one launch (a decode layer's gate | up or down), timed
                let mut tb6: Vec<u8> = Vec::new();
                for _ in 0..6 {
                    tb6.extend((src.ptr() as u64).to_le_bytes());
                }
                for i in 0..6 {
                    for v in [1i32, i, 0, 0, 0, 0, 0, 0, 0, 0] {
                        tb6.extend(v.to_le_bytes());
                    }
                }
                let tab6 = DevBuf::new(g, tb6.len()).map_err(e)?;
                tab6.write(0, &tb6).map_err(e)?;
                let y6 = DevBuf::f32(g, 6 * rows).map_err(e)?;
                o.std_mmv(t.ty.code(), &tab6, 6, &xd, (&y6, 0), 6, rows, cols).map_err(e)?;
                g.sync().map_err(e)?;
                let t0 = std::time::Instant::now();
                for _ in 0..10 {
                    o.std_mmv(t.ty.code(), &tab6, 6, &xd, (&y6, 0), 6, rows, cols).map_err(e)?;
                }
                g.sync().map_err(e)?;
                let us = t0.elapsed().as_secs_f64() * 1e5;
                println!("{tag:<28} [{rows} x {cols}] {}3-column product rel err {yrel:.2e}; 6 experts in one launch {us:.0} us", note);
            }
        }
    }
    Ok(())
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
    if m.g.std {
        return kernels_std(&f, &m, &g, &o);
    }
    // NS_DS41_IDX_BENCH=n1,n2,..: only the indexer's costs at a deep position - its scores for a prompt step's rows (as
    // many as select() takes at n keys), their top-k over all n, and the two-level form (blocks' maxima, the best
    // blocks, their keys scored again, the top-k among them) with whether its picks are the same
    if let Ok(v) = std::env::var("NS_DS41_IDX_BENCH") {
        let gg = &m.g;
        let (ih, id, kp) = (gg.idx_heads as usize, gg.idx_dim as usize, gg.idx_top_k as usize);
        let (bs, ct) = (gg.cand_block.max(1) as usize, gg.cand_top as usize);
        let r = gg.ratio(gg.index_sources[0]) as usize;
        for n in v.split(',').filter_map(|x| x.trim().parse::<usize>().ok()) {
            let t = ((32usize << 20) / n).clamp(1, 2048);
            let pos0 = n * r - t;
            let mut seed2 = 0x9E37_79B9u64 ^ n as u64;
            let mut rn = || {
                seed2 ^= seed2 >> 12;
                seed2 ^= seed2 << 25;
                seed2 ^= seed2 >> 27;
                ((seed2.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32 - 0.5
            };
            let q: Vec<f32> = (0..t * ih * id).map(|_| rn()).collect();
            let w: Vec<f32> = (0..t * ih).map(|_| rn()).collect();
            let keys: Vec<u8> = (0..n * id).flat_map(|_| f16_bits(rn()).to_le_bytes()).collect();
            let (qd, wd) = (DevBuf::from_f32(&g, &q).map_err(e)?, DevBuf::from_f32(&g, &w).map_err(e)?);
            let kd = DevBuf::new(&g, keys.len()).map_err(e)?;
            kd.write(0, &keys).map_err(e)?;
            let sc = DevBuf::f32(&g, t * n).map_err(e)?;
            let sel = DevBuf::new(&g, t * kp * 4).map_err(e)?;
            let nb = n.div_ceil(bs);
            let bmax = DevBuf::f32(&g, t * nb).map_err(e)?;
            let cand = DevBuf::new(&g, t * ct * 4).map_err(e)?;
            let scc = DevBuf::f32(&g, t * ct * bs).map_err(e)?;
            let selc = DevBuf::new(&g, t * kp * 4).map_err(e)?;
            let sel2 = DevBuf::new(&g, t * kp * 4).map_err(e)?;
            let time = |f: &dyn Fn() -> nextsycl_core::Result<()>| -> Result<f64, String> {
                f().map_err(e)?;
                g.sync().map_err(e)?;
                let t0 = std::time::Instant::now();
                for _ in 0..3 {
                    f().map_err(e)?;
                }
                g.sync().map_err(e)?;
                Ok(t0.elapsed().as_secs_f64() * 1e3 / 3.0)
            };
            let a = time(&|| o.idx_score(&qd, &wd, &kd, &sc, t, ih, id, n, n, pos0, r))?;
            let b = time(&|| o.topk(&sc, &sel, t, n, n, kp))?;
            let c = time(&|| {
                o.block_max(&sc, &bmax, t, n, n, bs, pos0, r)?;
                o.topk(&bmax, &cand, t, nb, nb, ct)?;
                o.idx_score_cand(&qd, &wd, &kd, &cand, ct, &scc, t, ih, id, n, pos0, r)?;
                o.topk(&scc, &selc, t, ct * bs, ct * bs, kp)?;
                o.cand_pos(&selc, &cand, &sel2, t, ct, kp, bs)
            })?;
            let (mut s1, mut s2) = (vec![0u8; t * kp * 4], vec![0u8; t * kp * 4]);
            sel.read(0, &mut s1).map_err(e)?;
            sel2.read(0, &mut s2).map_err(e)?;
            let same = s1.chunks(4 * kp).zip(s2.chunks(4 * kp)).filter(|(x, y)| x == y).count();
            println!("indexer at {n} keys, {t} rows: scores {a:.1} ms, top-{kp} {b:.1} ms; two-level (blocks of {bs}, best {ct}) {c:.1} ms, \
                      rows with the same picks {same} of {t}");
        }
        return Ok(());
    }
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
            o.vq_mmv(&tab, 1, (&cbd, None), &xd, (&yd, 0), 3, rows, cols, nbit, 0, None).map_err(e)?;
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
            // a prompt pass's product on XMX (ns_ds41_vq_gemm): 40 tokens of fp16 x, against the host's W x
            let nt: usize = std::env::var("NS_DS41_GEMM_TEST_N").ok().and_then(|v| v.parse().ok()).unwrap_or(40).clamp(1, 64);
            let xs: Vec<f32> = (0..64 * cols).map(|_| rnd()).collect();
            let xf = DevBuf::from_f32(&g, &xs).map_err(e)?;
            let xh = DevBuf::new(&g, xs.len() * 2).map_err(e)?;
            o.to_f16(&xf, &xh, xs.len()).map_err(e)?;
            let mut x16 = vec![0u8; xs.len() * 2];
            xh.read(0, &mut x16).map_err(e)?;
            let xr: Vec<f32> = x16.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
            let cbh = DevBuf::new(&g, cb.len() * 2).map_err(e)?;
            o.vq_codebook_f16(&cbd, cb.len(), &cbh).map_err(e)?;
            let yg = DevBuf::f32(&g, nt * rows).map_err(e)?;
            o.vq_gemm(&pd, None, rows, rows, cols, nbit, &cbh, &xh, cols, nt, (&yg, 0), rows).map_err(e)?;
            g.sync().map_err(e)?;
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                o.vq_gemm(&pd, None, rows, rows, cols, nbit, &cbh, &xh, cols, nt, (&yg, 0), rows).map_err(e)?;
            }
            g.sync().map_err(e)?;
            let gus = t0.elapsed().as_secs_f64() * 1e5;
            let got = yg.to_f32().map_err(e)?;
            let mut yw = vec![0f32; nt * rows];
            for c in 0..nt {
                for r in 0..rows {
                    yw[c * rows + r] = (0..cols).map(|k| want[r * cols + k] as f64 * xr[c * cols + k] as f64).sum::<f64>() as f32;
                }
            }
            let (grel, gmx) = compare(&got, &yw);
            println!("{:<26} vq_gemm {nt} tokens: rel err {grel:.2e} (max {gmx:.2e}), {gus:.0} us", "");
            if which == 0 {
                // gate and up as one matrix (two payloads, the split at the gate's rows), as a prompt pass runs them
                let mut pu = vec![0u8; len as usize];
                f.read_into(t, slot(1), &mut pu).map_err(|e| e.0)?;
                let wu = vq_host(&pu, &cb, rows, cols, nbit);
                let pud = DevBuf::new(&g, pu.len().next_multiple_of(256)).map_err(e)?;
                pud.write(0, &pu).map_err(e)?;
                let y2 = DevBuf::f32(&g, nt * 2 * rows).map_err(e)?;
                o.vq_gemm(&pd, Some(&pud), rows, 2 * rows, cols, nbit, &cbh, &xh, cols, nt, (&y2, 0), 2 * rows).map_err(e)?;
                let got = y2.to_f32().map_err(e)?;
                let mut yw = vec![0f32; nt * 2 * rows];
                for c in 0..nt {
                    for r in 0..2 * rows {
                        let w = if r < rows { &want[r * cols..(r + 1) * cols] } else { &wu[(r - rows) * cols..(r - rows + 1) * cols] };
                        yw[c * 2 * rows + r] = (0..cols).map(|k| w[k] as f64 * xr[c * cols + k] as f64).sum::<f64>() as f32;
                    }
                }
                let (grel, gmx) = compare(&got, &yw);
                println!("{:<26} vq_gemm gate | up {nt} tokens: rel err {grel:.2e} (max {gmx:.2e})", "");
            }
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
            // the lookup-table variant's lists: every gate | up reads x's one column, each down its own row of h
            tb.extend(crate::ops::lut_lists(1, (0..2 * n_e).map(|e| (0, e, 0))).ok_or("lists")?.0);
            db.extend(crate::ops::lut_lists(n_e, (0..n_e).map(|e| (e, e, 0))).ok_or("lists")?.0);
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
        for (name, tabs, n, x, y, rows, cols, lut) in [("gate | up", &gts, 2 * n_e, &xd, &gu, ff, d, (1, 2 * n_e)), ("down", &dts, n_e, &hd, &dn, d, ff, (n_e, 1))] {
            let mut first: Option<Vec<f32>> = None;
            for &v in &variants {
                let t0 = std::time::Instant::now();
                while t0.elapsed().as_secs_f64() < 0.5 {
                    for tab in tabs.iter() {
                        o.vq_mmv(tab, n, (&cbd, Some(if v >= 2 && v != 4 { &cbh } else { &cbf })), x, (y, 0), n, rows, cols, nbit, v, Some(lut)).map_err(e)?;
                    }
                    g.sync().map_err(e)?;
                }
                let it = 40;
                let t0 = std::time::Instant::now();
                for i in 0..it {
                    o.vq_mmv(&tabs[i % sets], n, (&cbd, Some(if v >= 2 && v != 4 { &cbh } else { &cbf })), x, (y, 0), n, rows, cols, nbit, v, Some(lut)).map_err(e)?;
                }
                g.sync().map_err(e)?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / it as f64;
                o.vq_mmv(&tabs[0], n, (&cbd, Some(if v >= 2 && v != 4 { &cbh } else { &cbf })), x, (y, 0), n, rows, cols, nbit, v, Some(lut)).map_err(e)?;
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
        // three columns (a verify pass of two drafts): its time, and its first column against the one-column product
        // (a verify pass's rows exactly as one-token passes)
        {
            let x3: Vec<f32> = x.iter().chain(&x).chain(&x).copied().collect();
            let x3d = DevBuf::from_f32(&g, &x3).map_err(e)?;
            let y3 = DevBuf::f32(&g, 3 * rows).map_err(e)?;
            o.q4k_mv(&wd, rows, cols, (&x3d, 0, cols), None, (&y3, 0, rows), 3).map_err(e)?;
            g.sync().map_err(e)?;
            let t0 = std::time::Instant::now();
            for _ in 0..it {
                o.q4k_mv(&wd, rows, cols, (&x3d, 0, cols), None, (&y3, 0, rows), 3).map_err(e)?;
            }
            g.sync().map_err(e)?;
            let us3 = t0.elapsed().as_secs_f64() * 1e6 / it as f64;
            let same = y3.view(0, rows * 4).map_err(e)?.to_f32().map_err(e)? == yd.to_f32().map_err(e)?;
            println!("q4k_mv {:<24} 3 columns: {us3:.1} us; column 0 {} the one-column product", t.name, if same { "bit-identical to" } else { "differs from" });
        }
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
        o.fp8_mmv(&wd, rows, cols, (&xd, 0, cols), None, (&yd, 0, rows), 2).map_err(e)?;
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
