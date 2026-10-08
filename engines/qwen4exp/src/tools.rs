//! Qwen3.8-Flash-Next's tools behind the command line's `info`.

use std::fmt::Write as _;

use ns_gguf::Gguf;

use crate::model::{Group, Model};

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// `nextsycl info`: the architecture and geometry, every tensor checked by role, bytes by group
pub fn info(f: &Gguf) -> Result<String, String> {
    let mut out = String::new();
    let _ = writeln!(out, "arch     : qwen4exp ({})", f.meta("general.name").and_then(|v| v.as_str()).unwrap_or("?"));
    let m = Model::open(f).map_err(|e| e.0)?;
    let g = &m.g;
    let qsa: Vec<u64> = (0..g.n_layer).filter(|l| g.is_qsa(*l)).collect();
    let _ = writeln!(out, "layers   : {} ({} Gated DeltaNet, {} QSA at {:?})", g.n_layer, g.n_layer - qsa.len() as u64, qsa.len(), qsa);
    let _ = writeln!(out, "width    : hidden {}, vocab {}, rms eps {:e}", g.n_embd, g.n_vocab, g.rms_eps);
    let _ = writeln!(out, "experts  : {} of {} per token, FFN {}; a shared expert of {}", g.n_expert_used, g.n_expert, g.ffn_expert, g.ffn_shared);
    let _ = writeln!(out, "QSA      : {} heads of {} (q | gate), {} KV heads, rotary {} dims base {}; indexer {} heads of {}, top {}, {} cells a block",
                     g.n_head, g.head_dim, g.n_head_kv, g.n_rot, g.rope_base, g.idx_heads, g.idx_dim, g.idx_top_k, g.idx_block);
    let _ = writeln!(out, "GDN      : {} k heads, {} v heads, state {}, conv {} over {} channels", g.gdn_k_heads, g.gdn_v_heads, g.gdn_state, g.gdn_conv,
                     g.gdn_channels());
    let _ = writeln!(out, "hc       : {} streams, rank {}", g.hc, g.hc_rank);
    let _ = writeln!(out, "PLE      : layers {:?}, {} rows of {} a token ({}-grams), conv {}, table {} rows", g.ple_layers, g.ple_heads(), g.ple_dim, g.ple_ngram,
                     g.ple_conv, g.ple_rows);
    let _ = writeln!(out, "MTP      : none in the file");

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
    let mut groups: std::collections::BTreeMap<Group, u64> = std::collections::BTreeMap::new();
    for ((grp, ty), (b, n)) in &m.bytes() {
        *groups.entry(*grp).or_default() += b;
        let _ = writeln!(out, "  {:<10} {:<8} {:>8.2} GiB  {:>4} tensors", format!("{grp:?}"), ty.name(), gib(*b), n);
    }
    for (grp, b) in &groups {
        let _ = writeln!(out, "  {:<19} {:>8.2} GiB", format!("{grp:?}"), gib(*b));
    }
    let e = (0..g.n_layer).map(|l| m.expert_bytes(l)).collect::<Vec<_>>();
    let (lo, hi) = (e.iter().min().copied().unwrap_or(0), e.iter().max().copied().unwrap_or(0));
    let _ = writeln!(out, "expert   : {:.2}-{:.2} MiB each (gate + up + down), {:.2}-{:.2} GiB per layer; a token reads {:.2} GiB of routed experts",
                     lo as f64 / 1048576.0, hi as f64 / 1048576.0, gib(lo * g.n_expert), gib(hi * g.n_expert),
                     gib(e.iter().sum::<u64>() * g.n_expert_used));
    if !errs.is_empty() {
        return Err(out);
    }
    Ok(out)
}
