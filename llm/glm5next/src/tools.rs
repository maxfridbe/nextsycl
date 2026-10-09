//! GLM-5.3's tools behind the command line's `info` and `kernels`.

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
    let _ = writeln!(out, "arch     : glm5-next ({})", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"));
    let m = Model::open(f).map_err(|e| e.0)?;
    let g = &m.g;
    let _ = writeln!(out, "scheme   : {:?} names", m.scheme);
    let _ = writeln!(out, "layers   : {} ({} dense, {} MoE; MLA at {:?}){}", g.n_layer, g.n_dense, g.n_layer - g.n_dense,
             (0..g.n_layer).filter(|l| g.is_mla(*l)).collect::<Vec<_>>(), if g.n_mtp > 0 { format!(" + {} MTP block", g.n_mtp) } else { " - no MTP block".into() });
    let _ = writeln!(out, "width    : hidden {}, vocab {}, dense FFN {}", g.n_embd, g.n_vocab, g.ffn_dense);
    let _ = writeln!(out, "experts  : {} of {} per token + {} shared, FFN {}, scale {} {}, SwiGLU limit {}", g.n_expert_used, g.n_expert, g.n_expert_shared, g.ffn_expert,
             g.expert_scale, if g.expert_norm { "(normalized)" } else { "" }, g.swiglu_limit);
    let _ = writeln!(out, "MLA      : {} heads of {}, q LoRA {}, kv LoRA {}; indexer {} heads of {}, top {}, pool {}", g.n_head, g.head_dim, g.q_lora, g.kv_lora,
             g.idx_heads, g.idx_dim, g.idx_top_k, g.idx_pool);
    let _ = writeln!(out, "KDA      : {} heads of {}, conv {}, gate rank {}, gate floor {}", g.kda_heads, g.kda_dim, g.kda_conv, g.kda_rank, g.kda_gate_low);
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
    let e = (g.n_dense..g.n_layer).map(|l| m.expert_bytes(l)).collect::<Vec<_>>();
    let (lo, hi) = (e.iter().min().copied().unwrap_or(0), e.iter().max().copied().unwrap_or(0));
    let _ = writeln!(out, "expert   : {:.1}-{:.1} MiB each (gate + up + down), {:.2}-{:.2} GiB per layer; a token reads {:.2} GiB of routed experts",
             lo as f64 / 1048576.0, hi as f64 / 1048576.0, gib(lo * g.n_expert), gib(hi * g.n_expert),
             gib(e.iter().sum::<u64>() * g.n_expert_used));
    if !errs.is_empty() {
        print!("{out}");
        return Err(format!("{} layout problem(s)", errs.len()));
    }
    Ok(out)
}


/// `nextsycl kernels`: each stored weight type of the file through the decode kernels against the exact path (expand
/// to float32, multiply), on a real matrix of that type; and the decode kernel's rate.
pub fn kernels(model: &Path, gpu: usize) -> Result<(), String> {
    use crate::ops::Ops;
    use nextsycl_core::DevBuf;
    let e = |x: nextsycl_core::Error| x.0;
    let f = Gguf::open(model).map_err(|e| e.0)?;
    let m = Model::open(&f).map_err(|e| e.0)?;
    let g = nextsycl_core::Gpu::open(gpu).map_err(e)?;
    let o = Ops::new(g.clone()).map_err(e)?;
    println!("gpu: {}", g.name);
    // one matrix per (type, role kind): routed experts' expert 0, else the whole matrix
    let mut picks: Vec<(String, nextsycl_gguf::GType, usize, usize, Vec<u8>)> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for l in 0..m.g.n_layer {
        for r in m.roles(l) {
            let Some(t) = m.tensor(l, r) else { continue };
            if t.shape.len() < 2 || t.ty == nextsycl_gguf::GType::F32 || !seen.insert((t.ty, matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown))) {
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
        let w = DevBuf::new(&g, bytes.len()).map_err(e)?;
        w.write(0, &bytes).map_err(e)?;
        let wf = DevBuf::f32(&g, rows * cols).map_err(e)?;
        o.dequant(ty.code(), &w, 0, bytes.len(), rows * cols, &wf).map_err(e)?;
        if !o.mmvq_supported(ty.code()) {
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
            o.mmvq(ty.code(), (&w, 0), bytes.len(), &q, (&y, 0), cols, rows, nc).map_err(e)?;
            let (_, rel, _) = compare(&y.to_f32().map_err(e)?, &yr.to_f32().map_err(e)?);
            errs.push(rel);
            if nc == 1 {
                g.sync().map_err(e)?;
                let n = 50;
                let t0 = std::time::Instant::now();
                for _ in 0..n {
                    o.mmvq(ty.code(), (&w, 0), bytes.len(), &q, (&y, 0), cols, rows, 1).map_err(e)?;
                }
                g.sync().map_err(e)?;
                rate = bytes.len() as f64 * n as f64 / t0.elapsed().as_secs_f64() / 1e9;
            }
        }
        println!("{name:<34} {:<8} {:>12} {:>10.2e} {:>10.2e} {:>8.0} GB/s", ty.name(), format!("{rows}x{cols}"), errs[0], errs[1], rate);
    }
    Ok(())
}

