//! DeepSeek-V4's tools behind the command line's `info` and `kernels`.

use std::fmt::Write as _;
use std::path::Path;

use nextsycl_gguf::Gguf;

use crate::model::{Group, Model, Role};

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// A tiny deterministic generator (xorshift64*)
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// IEEE half to float
fn f16_to_f32(h: u16) -> f32 {
    let (s, ex, m) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let bits = match (ex, m) {
        (0, 0) => s << 31,
        (0, _) => return if s == 1 { -(m as f32) * 2f32.powi(-24) } else { m as f32 * 2f32.powi(-24) },
        (31, _) => (s << 31) | (0xff << 23) | (m << 13),
        _ => (s << 31) | ((ex + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

fn compare(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut ab, mut aa, mut bb, mut dd, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
        dd += (x - y) * (x - y);
        mx = mx.max((x - y).abs());
    }
    (ab / (aa.sqrt() * bb.sqrt()).max(1e-30), (dd / bb.max(1e-30)).sqrt(), mx)
}

/// `nextsycl info`: the architecture and geometry, every tensor checked by role, bytes by group
pub fn info(f: &Gguf) -> Result<String, String> {
    let mut out = String::new();
    let _ = writeln!(out, "arch     : deepseek4 ({})", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"));
    let m = Model::open(f).map_err(|e| e.0)?;
    let g = &m.g;
    let by_ratio = |r: u64| (0..g.n_layer).filter(|l| g.ratio(*l) == r).collect::<Vec<_>>();
    let _ = writeln!(out, "layers   : {} - raw window only {:?}, CSA (ratio 4) {}, HCA (ratio 128) {}; hash-routed {}", g.n_layer, by_ratio(0), by_ratio(4).len(),
                     by_ratio(128).len(), g.hash_layers);
    let _ = writeln!(out, "width    : hidden {}, vocab {}", g.n_embd, g.n_vocab);
    let _ = writeln!(out, "attention: {} heads of {} (rope {}), q LoRA {}, window {}, output {} groups of LoRA {}; indexer {} heads of {}, top {}",
                     g.n_head, g.head_dim, g.rope_dim, g.q_lora, g.window, g.o_groups, g.o_lora, g.idx_heads, g.idx_dim, g.idx_top_k);
    let _ = writeln!(out, "rope     : base {} (layers 0, 1), compressed base {} with YaRN x{} from {}", g.rope_base, g.comp_rope_base, g.yarn_factor, g.yarn_orig_ctx);
    let _ = writeln!(out, "experts  : {} of {} per token + {} shared, FFN {}, scale {} {}, SwiGLU limit {}", g.n_expert_used, g.n_expert, g.n_expert_shared, g.ffn_expert,
                     g.expert_scale, if g.expert_norm { "(normalized)" } else { "" }, g.swiglu_limit);
    let _ = writeln!(out, "hc       : {} streams, {} Sinkhorn iterations, eps {:e}; rms eps {:e}", g.hc, g.hc_iters, g.hc_eps, g.rms_eps);
    let errs = m.check();
    if errs.is_empty() {
        let _ = writeln!(out, "check    : every tensor present with the shape the runtime needs, none left over");
    } else {
        let _ = writeln!(out, "check    : {} problem{}", errs.len(), if errs.len() > 1 { "s" } else { "" });
        for e in errs.iter().take(40) {
            let _ = writeln!(out, "  {e}");
        }
    }
    let _ = writeln!(out, "bytes    :");
    let by = m.bytes();
    let mut groups: std::collections::BTreeMap<Group, u64> = std::collections::BTreeMap::new();
    for ((grp, ty), (b, n)) in &by {
        *groups.entry(*grp).or_default() += b;
        let _ = writeln!(out, "  {:<10} {:<8} {:>8.2} GiB  {:>4} tensors", format!("{grp:?}"), ty.name(), gib(*b), n);
    }
    for (grp, b) in &groups {
        let _ = writeln!(out, "  {:<19} {:>8.2} GiB", format!("{grp:?}"), gib(*b));
    }
    let e = (0..g.n_layer).map(|l| m.expert_bytes(l)).collect::<Vec<_>>();
    let (lo, hi) = (e.iter().min().copied().unwrap_or(0), e.iter().max().copied().unwrap_or(0));
    let _ = writeln!(out, "expert   : {:.1}-{:.1} MiB each (gate + up + down); a token reads {:.2} GiB of routed experts", lo as f64 / 1048576.0, hi as f64 / 1048576.0,
                     gib(e.iter().sum::<u64>() * g.n_expert_used));
    if !errs.is_empty() {
        print!("{out}");
        return Err(format!("{} layout problem(s)", errs.len()));
    }
    Ok(out)
}

/// `nextsycl kernels`: each stored weight type of the file through the decode kernels against the exact path (expand
/// to float32, multiply), on a real matrix of that type - MXFP4 through this engine's own; and the decode kernel's rate.
pub fn kernels(model: &Path, gpu: usize) -> Result<(), String> {
    use crate::ops::Ops;
    use nextsycl_core::DevBuf;
    let e = |x: nextsycl_core::Error| x.0;
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let m = Model::open(&f).map_err(|e| e.0)?;
    let g = nextsycl_core::Gpu::open(gpu).map_err(e)?;
    let o = Ops::new(g.clone()).map_err(e)?;
    println!("gpu: {}", g.name);
    let mut picks: Vec<(String, nextsycl_gguf::GType, usize, usize, Vec<u8>)> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for l in 0..m.g.n_layer {
        for r in m.roles(l) {
            let Some(t) = m.tensor(l, r) else { continue };
            if t.shape.len() < 2 || matches!(t.ty, nextsycl_gguf::GType::F32 | nextsycl_gguf::GType::I32) || !seen.insert((t.ty, matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown))) {
                continue;
            }
            let (rows, cols) = (t.shape[t.shape.len() - 2] as usize, t.shape[t.shape.len() - 1] as usize);
            let bytes = t.ty.bytes((rows * cols) as u64).unwrap_or(0) as usize;
            let mut b = vec![0u8; bytes];
            f.read_into(t, 0, &mut b).map_err(|e| e.0)?;
            picks.push((t.name.clone(), t.ty, rows, cols, b));
        }
    }
    let mut rng = Rng(12345);
    println!("{:<34} {:<8} {:>12} {:>10} {:>10} {:>11}", "matrix", "type", "rows x cols", "1 col err", "4 col err", "1 col rate");
    for (name, ty, rows, cols, bytes) in picks {
        let mx = ty.code() == 39;
        let w = DevBuf::new(&g, bytes.len()).map_err(e)?;
        w.write(0, &bytes).map_err(e)?;
        let wf = DevBuf::f32(&g, rows * cols).map_err(e)?;
        if mx {
            o.mxfp4_dequant(&w, 0, rows * cols, &wf, false).map_err(e)?;
        } else if o.dequant(ty.code(), &w, 0, bytes.len(), rows * cols, &wf).is_err() {
            // the float32 expander lacks some IQ types (IQ3_XXS): the fp16 one's values, widened on the host
            let h = DevBuf::new(&g, rows * cols * 2).map_err(e)?;
            o.dequant_f16(ty.code(), &w, 0, bytes.len(), rows * cols, &h).map_err(e)?;
            let mut hb = vec![0u8; rows * cols * 2];
            h.read(0, &mut hb).map_err(e)?;
            let v: Vec<f32> = hb.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
            wf.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).map_err(e)?;
        }
        if !mx && !o.mmvq_supported(ty.code()) {
            println!("{name:<34} {:<8} {:>12} no decode kernel", ty.name(), format!("{rows}x{cols}"));
            continue;
        }
        let mut errs = Vec::new();
        let mut rate = 0.0;
        for nc in [1usize, 4] {
            let xs: Vec<f32> = (0..nc * cols).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            let x = DevBuf::from_f32(&g, &xs).map_err(e)?;
            let yr = DevBuf::f32(&g, nc * rows).map_err(e)?;
            o.gemm(nc, rows, cols, &x, &wf, &yr, false).map_err(e)?;
            let q = DevBuf::new(&g, o.q8_1_bytes(cols, nc)).map_err(e)?;
            o.quantize_q8_1((&x, 0), &q, cols, nc).map_err(e)?;
            let y = DevBuf::f32(&g, nc * rows).map_err(e)?;
            let run = |y: &DevBuf, n: usize| if mx { o.mxfp4_mmvq((&w, 0), &q, (y, 0), cols, rows, n) } else { o.mmvq(ty.code(), (&w, 0), bytes.len(), &q, (y, 0), cols, rows, n) };
            run(&y, nc).map_err(e)?;
            let (_, rel, _) = compare(&y.to_f32().map_err(e)?, &yr.to_f32().map_err(e)?);
            errs.push(rel);
            if nc == 1 {
                g.sync().map_err(e)?;
                let n = 50;
                let t0 = std::time::Instant::now();
                for _ in 0..n {
                    run(&y, 1).map_err(e)?;
                }
                g.sync().map_err(e)?;
                rate = bytes.len() as f64 * n as f64 / t0.elapsed().as_secs_f64() / 1e9;
            }
        }
        println!("{name:<34} {:<8} {:>12} {:>10.2e} {:>10.2e} {:>8.0} GB/s", ty.name(), format!("{rows}x{cols}"), errs[0], errs[1], rate);
    }
    multi_bench(&f, &m, &g, &o, std::env::var("NS_DS4_BENCH_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(5)).map_err(e)?;
    Ok(())
}

/// The decode MoE's two launches (gate | up, down) on a layer's first 6 experts from the file, at 1 and 4 columns:
/// the time a launch and the weight bytes it reads a second
fn multi_bench(f: &Gguf, m: &Model, g: &std::sync::Arc<nextsycl_core::Gpu>, o: &crate::ops::Ops, l: u64) -> nextsycl_core::Result<()> {
    use nextsycl_core::DevBuf;
    let (n, sets) = (6usize, 4usize); // 4 sets of 6 experts in turn: past the L2 cache
    let mut bufs = Vec::new();
    for r in [Role::ExpGate, Role::ExpUp, Role::ExpDown] {
        let Some(tn) = m.tensor(l, r) else { return Ok(()) };
        let (rows, cols) = (tn.shape[tn.shape.len() - 2] as usize, tn.shape[tn.shape.len() - 1] as usize);
        let eb = tn.ty.bytes((rows * cols) as u64).unwrap_or(0) as usize;
        let mut b = vec![0u8; eb * n * sets];
        f.read_into(tn, 0, &mut b).map_err(|e| nextsycl_core::Error(e.0))?;
        let w = DevBuf::new(g, b.len())?;
        w.write(0, &b)?;
        bufs.push((tn.ty, rows, cols, eb, w));
    }
    let (gt, f_, d, geb, gw) = (bufs[0].0, bufs[0].1, bufs[0].2, bufs[0].3, &bufs[0].4);
    let (ueb, uw) = (bufs[1].3, &bufs[1].4);
    let (dt, deb, dw) = (bufs[2].0, bufs[2].3, &bufs[2].4);
    if !o.multi_supported(gt.code(), d) || !o.multi_supported(dt.code(), f_) {
        println!("multi-expert: layer {l} ({} / {}) not the kernel's types", gt.name(), dt.name());
        return Ok(());
    }
    println!("multi-expert, layer {l}: {n} experts, gate | up {} {f_}x{d}, down {} {d}x{f_}", gt.name(), dt.name());
    let mut rng = Rng(777);
    for t in [1usize, 4] {
        // entries: each expert against one token (t = 4: experts spread over the tokens, as a verify pass)
        let tk: Vec<i32> = (0..n).map(|i| (i % t) as i32).collect();
        let (mut gtabs, mut dtabs) = (Vec::new(), Vec::new());
        for s in 0..sets {
            let mut tb = Vec::new();
            for i in s * n..(s + 1) * n {
                tb.extend((gw.ptr() as u64 + (i * geb) as u64).to_le_bytes());
            }
            for i in s * n..(s + 1) * n {
                tb.extend((uw.ptr() as u64 + (i * ueb) as u64).to_le_bytes());
            }
            for v in tk.iter().chain(&tk) {
                tb.extend(v.to_le_bytes());
            }
            let gtab = DevBuf::new(g, tb.len())?;
            gtab.write(0, &tb)?;
            gtabs.push(gtab);
            let mut db = Vec::new();
            for i in s * n..(s + 1) * n {
                db.extend((dw.ptr() as u64 + (i * deb) as u64).to_le_bytes());
            }
            for i in 0..n as i32 {
                db.extend(i.to_le_bytes());
            }
            let dtab = DevBuf::new(g, db.len())?;
            dtab.write(0, &db)?;
            dtabs.push(dtab);
        }
        let xs: Vec<f32> = (0..t * d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let x = DevBuf::from_f32(g, &xs)?;
        let xq = DevBuf::new(g, o.q8_bytes(d, t))?;
        o.quantize_q8((&x, 0), &xq, d, t)?;
        let gu = DevBuf::f32(g, 2 * n * f_)?;
        let hs: Vec<f32> = (0..n * f_).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let h = DevBuf::from_f32(g, &hs)?;
        let hq = DevBuf::new(g, o.q8_bytes(f_, n))?;
        o.quantize_q8((&h, 0), &hq, f_, n)?;
        let dn = DevBuf::f32(g, n * d)?;
        let reps = 200;
        let time = |run: &dyn Fn(usize) -> nextsycl_core::Result<()>| -> nextsycl_core::Result<f64> {
            run(0)?;
            g.sync()?;
            let t0 = std::time::Instant::now();
            for i in 0..reps {
                run(i % sets)?;
            }
            g.sync()?;
            Ok(t0.elapsed().as_secs_f64() / reps as f64)
        };
        let s1 = time(&|s| o.mmvq_multi(gt.code(), &gtabs[s], 2 * n, &xq, t, (&gu, 0), d, f_))?;
        let s2 = time(&|s| o.mmvq_multi(dt.code(), &dtabs[s], n, &hq, n, (&dn, 0), f_, d))?;
        println!("  {t} column(s): gate | up {:7.1} us {:5.0} GB/s   down {:7.1} us {:5.0} GB/s", s1 * 1e6, (n * (geb + ueb)) as f64 / s1 / 1e9, s2 * 1e6,
                 (n * deb) as f64 / s2 / 1e9);
    }
    Ok(())
}
