//! `nextsycl`: the command line.
//!
//!     nextsycl info <model.gguf>      the architecture, its geometry, every tensor checked by role, bytes by group

use std::path::Path;
use std::process::ExitCode;

use ns_gguf::Gguf;
use ns_model::glm5next::{Group, Model};

const USAGE: &str = "usage:
  nextsycl info <model.gguf>    the architecture and geometry, every tensor checked by role, bytes by group";

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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("info") if args.len() == 2 => info(Path::new(&args[1])),
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
