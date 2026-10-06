//! `nextsycl`: the command line.
//!
//!     nextsycl info <model.gguf>      the architecture, its geometry, every tensor checked by role, bytes by group
//!     nextsycl gpus                   each GPU in its own context: memory, copies and their rates, GPU to GPU, and
//!                                     that device memory costs no host RAM

use std::path::Path;
use std::process::ExitCode;

use ns_gguf::Gguf;
use ns_model::glm5next::{Group, Model};

const USAGE: &str = "usage:
  nextsycl info <model.gguf>    the architecture and geometry, every tensor checked by role, bytes by group
  nextsycl gpus                 each GPU in its own context: memory, copy rates, GPU to GPU, host RAM unaffected";

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn info(path: &Path) -> Result<(), String> {
    let f = Gguf::open(path).map_err(|e| e.0)?;
    println!("file     : {} ({} file{}, {:.2} GiB of tensors, {} tensors)", path.display(), f.paths.len(), if f.paths.len() > 1 { "s" } else { "" },
             gib(f.total_bytes()), f.tensors.len());
    let arch = ns_model::architecture(&f).map_err(|e| e.0)?;
    println!("arch     : {arch} ({})", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"));
    let m = Model::open(&f).map_err(|e| e.0)?;
    let g = &m.g;
    println!("scheme   : {:?} names", m.scheme);
    println!("layers   : {} ({} dense, {} MoE; MLA at {:?}){}", g.n_layer, g.n_dense, g.n_layer - g.n_dense,
             (0..g.n_layer).filter(|l| g.is_mla(*l)).collect::<Vec<_>>(), if g.n_mtp > 0 { format!(" + {} MTP block", g.n_mtp) } else { " - no MTP block".into() });
    println!("width    : hidden {}, vocab {}, dense FFN {}", g.n_embd, g.n_vocab, g.ffn_dense);
    println!("experts  : {} of {} per token + {} shared, FFN {}, scale {} {}, SwiGLU limit {}", g.n_expert_used, g.n_expert, g.n_expert_shared, g.ffn_expert,
             g.expert_scale, if g.expert_norm { "(normalized)" } else { "" }, g.swiglu_limit);
    println!("MLA      : {} heads of {}, q LoRA {}, kv LoRA {}; indexer {} heads of {}, top {}, pool {}", g.n_head, g.head_dim, g.q_lora, g.kv_lora,
             g.idx_heads, g.idx_dim, g.idx_top_k, g.idx_pool);
    println!("KDA      : {} heads of {}, conv {}, gate rank {}, gate floor {}", g.kda_heads, g.kda_dim, g.kda_conv, g.kda_rank, g.kda_gate_low);
    println!("hc       : {} streams, {} Sinkhorn iterations, eps {:e}; rms eps {:e}", g.hc, g.hc_iters, g.hc_eps, g.rms_eps);

    let errs = m.check();
    if errs.is_empty() {
        println!("check    : every tensor present with the shape the runtime needs, none left over");
    } else {
        println!("check    : {} problem{}", errs.len(), if errs.len() > 1 { "s" } else { "" });
        for e in errs.iter().take(40) {
            println!("  {e}");
        }
    }

    println!("bytes    :");
    let by = m.bytes();
    let mut groups: std::collections::BTreeMap<Group, u64> = std::collections::BTreeMap::new();
    for ((grp, ty), (b, n)) in &by {
        *groups.entry(*grp).or_default() += b;
        println!("  {:<10} {:<8} {:>8.2} GiB  {:>4} tensors", format!("{grp:?}"), ty.name(), gib(*b), n);
    }
    for (grp, b) in &groups {
        println!("  {:<19} {:>8.2} GiB", format!("{grp:?}"), gib(*b));
    }
    let e = (g.n_dense..g.n_layer).map(|l| m.expert_bytes(l)).collect::<Vec<_>>();
    let (lo, hi) = (e.iter().min().copied().unwrap_or(0), e.iter().max().copied().unwrap_or(0));
    println!("expert   : {:.1}-{:.1} MiB each (gate + up + down), {:.2}-{:.2} GiB per layer; a token reads {:.2} GiB of routed experts",
             lo as f64 / 1048576.0, hi as f64 / 1048576.0, gib(lo * g.n_expert), gib(hi * g.n_expert),
             gib(e.iter().sum::<u64>() * g.n_expert_used));
    if !errs.is_empty() {
        return Err(format!("{} layout problem(s)", errs.len()));
    }
    Ok(())
}

/// MemAvailable, bytes.
fn host_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map_or(0, |kb| kb * 1024)
}

fn gpus() -> Result<(), String> {
    use ns_core::{DevBuf, Gpu, HostBuf};
    use std::time::Instant;
    let e = |x: ns_core::Error| x.0;
    let list = ns_core::gpus().map_err(e)?;
    if list.is_empty() {
        return Err("no Level Zero GPU".into());
    }
    let mut open = Vec::new();
    for (i, name) in &list {
        let g = Gpu::open(*i).map_err(e)?;
        let (total, free) = g.memory().map_err(e)?;
        println!("GPU {i}: {name}, {:.1} GiB{}", gib(total), free.map_or(String::new(), |f| format!(" ({:.1} GiB free)", gib(f))));
        open.push(g);
    }
    // copies: 256 MiB through pinned memory, each way, checked
    const N: usize = 256 << 20;
    for g in &open {
        let d = DevBuf::new(g, N).map_err(e)?;
        let mut h = HostBuf::new(g, N).map_err(e)?;
        for (i, b) in h.as_mut_slice().iter_mut().enumerate() {
            *b = (i * 7 + g.index) as u8;
        }
        let t = Instant::now();
        d.write(0, h.as_slice()).map_err(e)?;
        g.sync().map_err(e)?;
        let up = N as f64 / t.elapsed().as_secs_f64() / 1e9;
        let mut back = HostBuf::new(g, N).map_err(e)?;
        let t = Instant::now();
        d.read(0, back.as_mut_slice()).map_err(e)?;
        let down = N as f64 / t.elapsed().as_secs_f64() / 1e9;
        let ok = back.as_slice() == h.as_slice();
        println!("GPU {}: 256 MiB to the GPU {up:.1} GB/s, back {down:.1} GB/s, {}", g.index, if ok { "identical" } else { "DIFFERENT" });
        if !ok {
            return Err(format!("GPU {}: the round trip changed the data", g.index));
        }
    }
    // GPU to GPU, through host memory
    if open.len() > 1 {
        let (a, b) = (&open[0], &open[1]);
        let (src, dst) = (DevBuf::new(a, N).map_err(e)?, DevBuf::new(b, N).map_err(e)?);
        let mut h = HostBuf::new(a, N).map_err(e)?;
        for (i, x) in h.as_mut_slice().iter_mut().enumerate() {
            *x = (i * 13) as u8;
        }
        src.write(0, h.as_slice()).map_err(e)?;
        a.sync().map_err(e)?;
        let mut staging = vec![0u8; N];
        let t = Instant::now();
        dst.copy_from_peer(&src, &mut staging).map_err(e)?;
        let rate = N as f64 / t.elapsed().as_secs_f64() / 1e9;
        let mut back = vec![0u8; N];
        dst.read(0, &mut back).map_err(e)?;
        let ok = back == h.as_slice();
        println!("GPU 0 -> GPU 1: 256 MiB through host memory {rate:.1} GB/s, {}", if ok { "identical" } else { "DIFFERENT" });
        if !ok {
            return Err("the GPU-to-GPU copy changed the data".into());
        }
    }
    // device memory must not cost host memory (a context spanning both GPUs made xe mirror it); 1 GiB a card,
    // small enough beside a loaded model (a card past its memory spills, and on this box that can livelock)
    let before = host_available();
    let held: Vec<DevBuf> = open.iter().map(|g| {
        let b = DevBuf::new(g, 1 << 30)?;
        b.fill(1)?;
        g.sync()?;
        Ok(b)
    }).collect::<Result<_, ns_core::Error>>().map_err(e)?;
    std::thread::sleep(std::time::Duration::from_millis(500));
    let after = host_available();
    let used = before.saturating_sub(after);
    println!("{} GiB of device memory (1 a card) took {:.2} GiB of host memory{}", held.len(), gib(used),
             if used < 1 << 29 { " - no mirror" } else { " - MIRRORED" });
    if used >= 1 << 29 {
        return Err("device memory is being mirrored into host RAM".into());
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("info") if args.len() == 2 => info(Path::new(&args[1])),
        Some("gpus") => gpus(),
        _ => Err(USAGE.into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nextsycl: {e}");
            ExitCode::FAILURE
        }
    }
}
