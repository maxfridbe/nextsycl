//! DeepSeek-V3 / R1's tools behind the command line's `info` and `kernels`.

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

fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut ab, mut aa, mut bb, mut dd) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
        dd += (x - y) * (x - y);
    }
    (ab / (aa.sqrt() * bb.sqrt()).max(1e-30), (dd / bb.max(1e-30)).sqrt())
}

/// `nextsycl info`: the architecture and geometry, every tensor checked by role, bytes by group
pub fn info(f: &Gguf) -> Result<String, String> {
    let mut out = String::new();
    let _ = writeln!(out, "arch     : deepseek2 ({})", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"));
    let m = Model::open(f).map_err(|e| e.0)?;
    let g = &m.g;
    let _ = writeln!(out, "layers   : {} ({} dense, {} MoE); kv_b {}", g.n_layer, g.n_dense, g.n_layer - g.n_dense, if m.split_kv { "split (k_b, v_b)" } else { "whole" });
    let _ = writeln!(out, "width    : hidden {}, vocab {}, dense FFN {}", g.n_embd, g.n_vocab, g.ffn_dense);
    let _ = writeln!(out, "experts  : {} of {} a token + {} shared, FFN {}, scale {} {}", g.n_expert_used, g.n_expert, g.n_expert_shared, g.ffn_expert, g.expert_scale,
                     if g.expert_norm { "(normalized)" } else { "" });
    let (fs, ext, ms, c0, c1, kq) = m.yarn();
    let _ = writeln!(out, "MLA      : {} heads, key {} (rope {}), value {}, q LoRA {}, kv LoRA {}; YaRN x{} from {} (freq scale {fs}, ext {ext}, cos/sin x {ms}, corr {c0}-{c1}), scores x {kq}",
                     g.n_head, g.key_len, g.rope_dim, g.value_len, g.q_lora, g.kv_lora, g.yarn_factor, g.yarn_orig_ctx);
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
    let e = (g.n_dense..g.n_layer).map(|l| m.expert_bytes(l)).collect::<Vec<_>>();
    let (lo, hi) = (e.iter().min().copied().unwrap_or(0), e.iter().max().copied().unwrap_or(0));
    let _ = writeln!(out, "expert   : {:.1}-{:.1} MiB each (gate + up + down); a token reads {:.2} GiB of routed experts", lo as f64 / 1048576.0, hi as f64 / 1048576.0,
                     gib(e.iter().sum::<u64>() * g.n_expert_used));
    if !errs.is_empty() {
        print!("{out}");
        return Err(format!("{} layout problem(s)", errs.len()));
    }
    Ok(out)
}

/// `nextsycl kernels`: the experts' types through this engine's decode kernels (one matrix, and a layer's experts in
/// one launch) against the exact path (expand to float32, multiply), on the file's own experts; and their rates.
pub fn kernels(model: &Path, gpu: usize) -> Result<(), String> {
    use crate::ops::Ops;
    use nextsycl_core::DevBuf;
    let e = |x: nextsycl_core::Error| x.0;
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let m = Model::open(&f).map_err(|e| e.0)?;
    let g = nextsycl_core::Gpu::open(gpu).map_err(e)?;
    let o = Ops::new(g.clone()).map_err(e)?;
    println!("gpu: {}", g.name);
    let mut rng = Rng(12345);
    let mut seen = std::collections::BTreeSet::new();
    println!("{:<34} {:<8} {:>12} {:>10} {:>10} {:>10} {:>11}", "matrix (expert 0..7)", "type", "rows x cols", "1 col err", "4 col err", "multi err", "1 col rate");
    for l in m.g.n_dense..m.g.n_layer {
        for r in [Role::ExpGate, Role::ExpDown] {
            let Some(t) = m.tensor(l, r) else { continue };
            if !seen.insert(t.ty) {
                continue;
            }
            let (rows, cols) = (t.shape[1] as usize, t.shape[2] as usize);
            let per = t.ty.bytes((rows * cols) as u64).unwrap_or(0) as usize;
            let ne = 8;
            let mut b = vec![0u8; per * ne];
            f.read_into(t, 0, &mut b).map_err(|e| e.0)?;
            let w = DevBuf::new(&g, b.len() + 256).map_err(e)?; // the kernels' 16-byte loads may read past the last block
            w.write(0, &b).map_err(e)?;
            let wf = DevBuf::f32(&g, ne * rows * cols).map_err(e)?;
            o.dequant(t.ty.code(), &w, 0, b.len(), ne * rows * cols, &wf).map_err(e)?;
            let mut errs = Vec::new();
            for nc in [1usize, 4] {
                let xs: Vec<f32> = (0..nc * cols).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
                let x = DevBuf::from_f32(&g, &xs).map_err(e)?;
                let yr = DevBuf::f32(&g, nc * rows).map_err(e)?;
                o.gemm_at(nc, rows, cols, (&x, 0, cols), (&wf, 0), (&yr, 0, rows), false).map_err(e)?;
                let q = DevBuf::new(&g, o.q8_bytes(cols, nc)).map_err(e)?;
                o.quantize_q8((&x, 0), &q, cols, nc).map_err(e)?;
                let y = DevBuf::f32(&g, nc * rows).map_err(e)?;
                o.own_mmvq(t.ty.code(), (&w, 0), per, &q, (&y, 0), cols, rows, nc).map_err(e)?;
                errs.push(compare(&y.to_f32().map_err(e)?, &yr.to_f32().map_err(e)?).1);
            }
            // the 8 experts at once, each against its own column
            let xs: Vec<f32> = (0..ne * cols).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            let x = DevBuf::from_f32(&g, &xs).map_err(e)?;
            let yr = DevBuf::f32(&g, ne * rows).map_err(e)?;
            for k in 0..ne {
                o.gemm_at(1, rows, cols, (&x, k * cols, cols), (&wf, k * rows * cols), (&yr, k * rows, rows), false).map_err(e)?;
            }
            let q = DevBuf::new(&g, o.q8_bytes(cols, ne)).map_err(e)?;
            o.quantize_q8((&x, 0), &q, cols, ne).map_err(e)?;
            let mut tb: Vec<u8> = (0..ne).flat_map(|k| (w.ptr() as u64 + (k * per) as u64).to_le_bytes()).collect();
            tb.extend((0..ne as i32).flat_map(|k| k.to_le_bytes()));
            let table = DevBuf::new(&g, tb.len()).map_err(e)?;
            table.write(0, &tb).map_err(e)?;
            let y = DevBuf::f32(&g, ne * rows).map_err(e)?;
            o.mmvq_multi(t.ty.code(), &table, ne, &q, ne, &y, cols, rows).map_err(e)?;
            let multi = compare(&y.to_f32().map_err(e)?, &yr.to_f32().map_err(e)?).1;
            g.sync().map_err(e)?;
            let reps = 50;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                o.mmvq_multi(t.ty.code(), &table, ne, &q, ne, &y, cols, rows).map_err(e)?;
            }
            g.sync().map_err(e)?;
            let multi_rate = (per * ne) as f64 * reps as f64 / t0.elapsed().as_secs_f64() / 1e9;
            // the same while the copy engine streams slots up (the store's swaps beside the experts' compute)
            {
                let slot = per * 3 + 4096;
                let nc = 16;
                let host = nextsycl_core::HostBuf::new(&g, nc * slot).map_err(e)?;
                let dev = DevBuf::new(&g, nc * slot).map_err(e)?;
                let hv = host.device_view(0, nc * slot).map_err(e)?;
                g.sync().map_err(e)?;
                let t0 = std::time::Instant::now();
                let mut last = 0;
                for i in 0..nc {
                    last = o.stream_copy_on(1, &dev, i * slot, &hv, i * slot, slot, &[]).map_err(e)?;
                }
                let tq = std::time::Instant::now();
                for _ in 0..reps {
                    o.mmvq_multi(t.ty.code(), &table, ne, &q, ne, &y, cols, rows).map_err(e)?;
                }
                g.sync().map_err(e)?;
                let tk = tq.elapsed().as_secs_f64();
                o.wait_ticket(last).map_err(e)?;
                println!("  {} x 8 experts with {nc} slot copies up beside them: {:.0} GB/s ({:.1} ms; copies done after {:.1} ms)", reps,
                         (per * ne) as f64 * reps as f64 / tk / 1e9, tk * 1e3, t0.elapsed().as_secs_f64() * 1e3);
            }
            // the rate: one column, one expert, 50 times
            let q1 = DevBuf::new(&g, o.q8_bytes(cols, 1)).map_err(e)?;
            o.quantize_q8((&x, 0), &q1, cols, 1).map_err(e)?;
            g.sync().map_err(e)?;
            let n = 50;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                o.own_mmvq(t.ty.code(), (&w, 0), per, &q1, (&y, 0), cols, rows, 1).map_err(e)?;
            }
            g.sync().map_err(e)?;
            let rate = per as f64 * n as f64 / t0.elapsed().as_secs_f64() / 1e9;
            println!("{:<34} {:<8} {:>12} {:>10.2e} {:>10.2e} {:>10.2e} {:>8.0} GB/s {:>8.0} GB/s (8 at once)", t.name, t.ty.name(), format!("{rows}x{cols}"), errs[0], errs[1], multi, rate, multi_rate);
        }
    }
    // the dense types this engine multiplies itself (Q4_K, Q5_K) against the shared kernels: MLA's output projection,
    // the shared expert's gate
    for (l, r) in [(10u64, Role::Out), (10u64, Role::QB), (10u64, Role::ShGate)] {
        let t = m.tensor(l, r).ok_or("tensor missing")?;
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        let mut b = vec![0u8; t.bytes as usize];
        f.read_into(t, 0, &mut b).map_err(|e| e.0)?;
        let w = DevBuf::new(&g, b.len() + 256).map_err(e)?;
        w.write(0, &b).map_err(e)?;
        let xv: Vec<f32> = (0..cols).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let x = DevBuf::from_f32(&g, &xv).map_err(e)?;
        let q1 = DevBuf::new(&g, o.q8_1_bytes(cols, 1)).map_err(e)?;
        o.quantize_q8_1((&x, 0), &q1, cols, 1).map_err(e)?;
        let qs = DevBuf::new(&g, o.q8_bytes(cols, 1)).map_err(e)?;
        o.quantize_q8((&x, 0), &qs, cols, 1).map_err(e)?;
        let (ya, yb) = (DevBuf::f32(&g, rows).map_err(e)?, DevBuf::f32(&g, rows).map_err(e)?);
        o.mmvq(t.ty.code(), (&w, 0), b.len(), &q1, (&ya, 0), cols, rows, 1).map_err(e)?;
        o.own_mmvq(t.ty.code(), (&w, 0), b.len(), &qs, (&yb, 0), cols, rows, 1).map_err(e)?;
        let (cos, rel) = compare(&yb.to_f32().map_err(e)?, &ya.to_f32().map_err(e)?);
        let rate = |own: bool| -> Result<f64, String> {
            g.sync().map_err(e)?;
            let t0 = std::time::Instant::now();
            for _ in 0..50 {
                if own {
                    o.own_mmvq(t.ty.code(), (&w, 0), b.len(), &qs, (&yb, 0), cols, rows, 1).map_err(e)?;
                } else {
                    o.mmvq(t.ty.code(), (&w, 0), b.len(), &q1, (&ya, 0), cols, rows, 1).map_err(e)?;
                }
            }
            g.sync().map_err(e)?;
            Ok(b.len() as f64 * 50.0 / t0.elapsed().as_secs_f64() / 1e9)
        };
        println!("{:<34} {:<8} {:>12} own vs shared: cosine {cos:.7}, rel {rel:.2e}; {:.0} vs {:.0} GB/s", t.name, t.ty.name(), format!("{rows}x{cols}"), rate(true)?, rate(false)?);
    }
    // the CPU's products (cpu.rs) against the GPU's on layer 10's experts 0..7: one matrix's rows, then whole experts
    {
        let l = 10;
        let pg = m.tensor(l, Role::ExpGate).ok_or("no experts")?;
        let pd = m.tensor(l, Role::ExpDown).ok_or("no experts")?;
        let (ff, dd) = (m.g.ffn_expert as usize, m.g.n_embd as usize);
        let per = (pg.bytes / m.g.n_expert) as usize;
        let perd = (pd.bytes / m.g.n_expert) as usize;
        let ne = 8;
        // slots as the store lays them out: [gate | up | down]
        let slot = 2 * per + perd;
        let mut slots = vec![0u8; ne * slot + 256];
        let pu = m.tensor(l, Role::ExpUp).ok_or("no experts")?;
        for k in 0..ne {
            let b = k * slot;
            f.read_into(pg, (k * per) as u64, &mut slots[b..b + per]).map_err(|e| e.0)?;
            f.read_into(pu, (k * per) as u64, &mut slots[b + per..b + 2 * per]).map_err(|e| e.0)?;
            f.read_into(pd, (k * perd) as u64, &mut slots[b + 2 * per..b + slot]).map_err(|e| e.0)?;
        }
        let xs: Vec<f32> = (0..dd).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        // GPU: expert 0's gate on the same activations
        let w = DevBuf::new(&g, slots.len()).map_err(e)?;
        w.write(0, &slots).map_err(e)?;
        let x = DevBuf::from_f32(&g, &xs).map_err(e)?;
        let q = DevBuf::new(&g, o.q8_bytes(dd, 1)).map_err(e)?;
        o.quantize_q8((&x, 0), &q, dd, 1).map_err(e)?;
        let y = DevBuf::f32(&g, ff).map_err(e)?;
        o.own_mmvq(19, (&w, 0), per, &q, (&y, 0), dd, ff, 1).map_err(e)?;
        let gpu_y = y.to_f32().map_err(e)?;
        let xq = crate::cpu::quantize(&xs);
        let mut cpu_y = vec![0f32; ff];
        crate::cpu::matvec(&slots[..per], dd, &xq, 0..ff, &mut cpu_y);
        let (cos, rel) = compare(&cpu_y, &gpu_y);
        println!("cpu IQ1_S gate (layer {l} expert 0) against the GPU's: cosine {cos:.7}, rel err {rel:.2e}");
        let threads: usize = std::env::var("NS_DS2_CPU_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        let pool = crate::cpu::Pool::new(threads);
        let jobs: Vec<crate::cpu::Job> = (0..ne).map(|k| crate::cpu::Job { slot: &slots[k * slot..(k + 1) * slot], x: &xq }).collect();
        let _ = crate::cpu::experts(&pool, &jobs, dd, ff, 2 * per);
        let reps = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let _ = crate::cpu::experts(&pool, &jobs, dd, ff, 2 * per);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        println!("cpu: {ne} whole experts (gate | up, SwiGLU, down) on {threads} threads: {:.2} ms ({:.1} GB/s of weights)", dt * 1e3, (ne * slot) as f64 / dt / 1e9);
    }
    // the expert store's copies: 32 slot-sized copies pinned memory -> VRAM (lane 1), VRAM -> pinned (lane 0), both
    let slot = 8_601_600usize;
    let n = 32;
    let mut host = nextsycl_core::HostBuf::new(&g, n * slot).map_err(e)?;
    host.as_mut_slice().iter_mut().enumerate().for_each(|(i, b)| *b = (i * 7) as u8);
    let dev = DevBuf::new(&g, n * slot).map_err(e)?;
    let dev2 = DevBuf::new(&g, n * slot).map_err(e)?;
    let hv = host.device_view(0, n * slot).map_err(e)?;
    for (name, up, down, kernel) in [("pinned -> VRAM", true, false, false), ("VRAM -> pinned", false, true, false), ("both at once", true, true, false),
                                     ("VRAM -> pinned (kernel)", false, true, true), ("up + kernel down", true, true, true)] {
        g.sync().map_err(e)?;
        let t0 = std::time::Instant::now();
        let mut last = Vec::new();
        for i in 0..n {
            if up {
                last.push(o.stream_copy_on(1, &dev, i * slot, &hv, i * slot, slot, &[]).map_err(e)?);
            }
            if down && kernel {
                last.push(o.kcopy(&hv, i * slot, &dev2, i * slot, slot, None).map_err(e)?);
            } else if down {
                last.push(o.stream_copy_on(0, &hv, i * slot, &dev2, i * slot, slot, &[]).map_err(e)?);
            }
        }
        for t in last {
            o.wait_ticket(t).map_err(e)?;
        }
        let secs = t0.elapsed().as_secs_f64();
        let bytes = (n * slot * (up as usize + down as usize)) as f64;
        println!("copies {name:<24}: {:.1} GB/s ({} of {:.1} MB in {:.1} ms)", bytes / secs / 1e9, n * (up as usize + down as usize), slot as f64 / 1e6, secs * 1e3);
    }
    Ok(())
}
