//! MiniMax-M3.1-Flash's geometry (`minimax-m3.*` metadata) and its tensors by role, checked against a file's shapes.
//! A split GGUF keeps the metadata in shard 1 (no tensors) and the tensors in the shards after it: open shard 1 with
//! the others as extras (`Gguf::open_with`).

use nextsycl_gguf::{GType, Gguf};

use crate::{Error, ModelResult as Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub n_embd: u64,
    pub n_vocab: u64,
    pub n_layer: u64,
    /// the first layers with a dense feed-forward (`ffn_{gate,up,down}`), the rest MoE
    pub n_dense: u64,
    pub n_head: u64,
    pub n_kv_head: u64,
    pub head_dim: u64,
    /// the dimensions of a head that are rotated (the first `n_rot`)
    pub n_rot: u64,
    pub rope_base: f64,
    pub rms_eps: f64,
    pub n_ff: u64,
    pub n_expert: u64,
    pub n_expert_used: u64,
    pub n_ff_expert: u64,
    pub n_shared: u64,
    pub expert_scale: f32,
    pub expert_norm: bool,
    /// 2 = sigmoid (llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID); the router adds `exp_probs_b` to pick, not to weigh
    pub gating: u64,
    pub indexer: Indexer,
}

/// MiniMax Sparse Attention's block indexer: a query scores the keys' blocks with a small head set of its own and
/// attends only the `top_k` best blocks plus the `local_blocks` nearest
#[derive(Clone, Debug, PartialEq)]
pub struct Indexer {
    pub n_head: u64,
    pub key_len: u64,
    pub top_k: u64,
    pub block_size: u64,
    pub local_blocks: u64,
}

impl Geometry {
    pub fn from_gguf(g: &Gguf) -> Result<Geometry> {
        if g.architecture() != "minimax-m3" {
            return Err(Error(format!("architecture {}: minimax-m3 expected", g.architecture())));
        }
        let u = |k: &str| g.arch_meta(k).and_then(|v| v.as_u64()).ok_or_else(|| Error(format!("minimax-m3.{k}: missing")));
        let f = |k: &str| g.arch_meta(k).and_then(|v| v.as_f64()).ok_or_else(|| Error(format!("minimax-m3.{k}: missing")));
        let n_vocab = g.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error("token_embd.weight: missing".into()))?;
        Ok(Geometry {
            n_embd: u("embedding_length")?,
            n_vocab,
            n_layer: u("block_count")?,
            n_dense: u("leading_dense_block_count")?,
            n_head: u("attention.head_count")?,
            n_kv_head: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            rope_base: f("rope.freq_base")?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            n_ff: u("feed_forward_length")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_expert: u("expert_feed_forward_length")?,
            n_shared: u("expert_shared_count")?,
            expert_scale: f("expert_weights_scale")? as f32,
            expert_norm: g.arch_meta("expert_weights_norm").map(|v| v.as_u64() == Some(1)).unwrap_or(false),
            gating: u("expert_gating_func")?,
            indexer: Indexer {
                n_head: u("attention.indexer.head_count")?,
                key_len: u("attention.indexer.key_length")?,
                top_k: u("attention.indexer.top_k")?,
                block_size: u("attention.indexer.block_size")?,
                local_blocks: u("attention.indexer.local_blocks")?,
            },
        })
    }

    /// The tensors one layer has, with the shape each must have (ggml order: the row length first)
    pub fn layer_tensors(&self, l: u64) -> Vec<(String, Vec<u64>)> {
        let (e, hd) = (self.n_embd, self.head_dim);
        let n = |s: &str| format!("blk.{l}.{s}.weight");
        let mut t = vec![
            (n("attn_norm"), vec![e]),
            (n("attn_q"), vec![e, self.n_head * hd]),
            (n("attn_k"), vec![e, self.n_kv_head * hd]),
            (n("attn_v"), vec![e, self.n_kv_head * hd]),
            (n("attn_q_norm"), vec![hd]),
            (n("attn_k_norm"), vec![hd]),
            (n("attn_output"), vec![self.n_head * hd, e]),
            (n("indexer.q_proj"), vec![e, self.indexer.n_head * self.indexer.key_len]),
            (n("indexer.k_proj"), vec![e, self.indexer.key_len]),
            (n("indexer.q_norm"), vec![self.indexer.key_len]),
            (n("indexer.k_norm"), vec![self.indexer.key_len]),
            (n("ffn_norm"), vec![e]),
        ];
        if l < self.n_dense {
            t.extend([(n("ffn_gate"), vec![e, self.n_ff]), (n("ffn_up"), vec![e, self.n_ff]), (n("ffn_down"), vec![self.n_ff, e])]);
        } else {
            let (x, fe) = (self.n_expert, self.n_ff_expert);
            t.extend([
                (n("ffn_gate_inp"), vec![e, x]),
                (format!("blk.{l}.exp_probs_b.bias"), vec![x]),
                (n("ffn_gate_exps"), vec![e, fe, x]),
                (n("ffn_up_exps"), vec![e, fe, x]),
                (n("ffn_down_exps"), vec![fe, e, x]),
                (n("ffn_gate_shexp"), vec![e, fe]),
                (n("ffn_up_shexp"), vec![e, fe]),
                (n("ffn_down_shexp"), vec![fe, e]),
            ]);
        }
        t
    }

    /// Every tensor of the model checked against the file: present, the shape, and a type the engine has a kernel for
    pub fn check(&self, g: &Gguf) -> Result<()> {
        let mut all = vec![("token_embd.weight".to_string(), vec![self.n_embd, self.n_vocab]), ("output_norm.weight".into(), vec![self.n_embd]),
                           ("output.weight".into(), vec![self.n_embd, self.n_vocab])];
        for l in 0..self.n_layer {
            all.extend(self.layer_tensors(l));
        }
        for (name, dims) in all {
            let t = g.tensor(&name).ok_or_else(|| Error(format!("{name}: missing")))?;
            // the reader's shapes are outermost first, ggml's (these) innermost first
            let have: Vec<u64> = t.shape.iter().rev().copied().collect();
            if have != dims {
                return Err(Error(format!("{name}: shape {have:?}, {dims:?} expected")));
            }
            if !SUPPORTED.contains(&t.ty) {
                return Err(Error(format!("{name}: type {} has no kernel yet", t.ty.name())));
            }
        }
        Ok(())
    }
}

/// The row types UD-IQ1_M uses (experts IQ1_M gate/up, IQ3_XXS down; the rest Q4_K, Q5_K, Q6_K, F32); the other quants
/// of the repo add IQ2_XXS, Q2_K, Q3_K, IQ3_S, Q4_K, Q8_0
pub const SUPPORTED: &[GType] = &[GType::F32, GType::Q4K, GType::Q5K, GType::Q6K, GType::Q8_0, GType::IQ1M, GType::IQ2XXS, GType::IQ3XXS, GType::IQ4XS];

#[cfg(test)]
mod tests {
    use super::*;

    /// The real file, when NS_MM31_FILE names its first shard (the others beside it)
    #[test]
    fn the_file_matches_the_description() {
        let Ok(p) = std::env::var("NS_MM31_FILE") else { return };
        let p = std::path::PathBuf::from(p);
        let dir = p.parent().unwrap();
        let mut extra: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|x| x.extension().is_some_and(|e| e == "gguf") && *x != p).collect();
        extra.sort();
        let g = Gguf::open_with(&p, &extra).unwrap();
        let geo = Geometry::from_gguf(&g).unwrap();
        assert_eq!((geo.n_layer, geo.n_embd, geo.n_expert, geo.n_expert_used, geo.n_dense), (60, 6144, 128, 4, 3));
        geo.check(&g).unwrap();
    }
}
