//! Qwen3.8-Flash-Next (`qwen4exp`): 48 layers, hidden 2560, no dense feed-forward.
//!
//! ```text
//!   residual:      hyper-connections, 4 streams of 2560: each half-layer reads a mix of them (a per-stream RMS
//!                  norm, a rank-320 down / up gate, silu / sigmoid) and writes its output back into each stream
//!                  through its own sigmoid inject
//!   attention:     Gated DeltaNet (3 of every 4 layers): qkv [16 k/q heads of 128 | 48 v heads of 128] through a
//!                  4-tap causal conv + SiLU, L2-normed q / k, a softplus decay and a sigmoid beta per v head,
//!                  a per-head RMS norm gated by sigmoid(z); or
//!                  QSA (every 4th: layers 3, 7, ..., 47): gated full attention, 24 query heads of 256 (each [q |
//!                  gate]), 2 KV heads, RMS-normed q / k, NEOX rotary on 64 dims; an indexer (4 heads of 128, keys
//!                  pooled 4 cells a block) picks the top 2048 cells each query reads
//!   feed-forward:  MoE every layer: 512 experts of 640, top 10 by softmax (renormalized), plus one shared expert
//!                  of 640 scaled by sigmoid(gate_inp_shexp . x)
//!   PLE:           layer 1 only: an n-gram-hashed per-layer embedding (16 rows of 160 a token from a 320M-row table,
//!                  2- and 3-grams of the previous tokens), gated by a key / query score, convolved over 9 rows of
//!                  history, added to the residual streams
//!   head:          a last hyper-connection mix (no inject), then output [vocab, 2560]
//! ```
//!
//! The MTP draft layer is not in the GGUF (Strata fetches it from the base checkpoint), so this engine has none yet.

use ns_gguf::{GType, Gguf, Tensor, Value};

use crate::{Error, ModelResult as Result};

/// The geometry, from metadata and the tensors' shapes.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub n_embd: u64,
    pub n_vocab: u64,
    pub n_layer: u64,
    pub rms_eps: f64,
    /// MoE
    pub n_expert: u64,
    pub n_expert_used: u64,
    pub ffn_expert: u64,
    pub ffn_shared: u64,
    /// QSA (the full-attention layers)
    pub n_head: u64,
    pub n_head_kv: u64,
    pub head_dim: u64,
    pub n_rot: u64,
    pub rope_base: f64,
    pub idx_heads: u64,
    pub idx_dim: u64,
    pub idx_top_k: u64,
    /// cells a pooled indexer block covers
    pub idx_block: u64,
    /// Gated DeltaNet
    pub gdn_state: u64,
    pub gdn_k_heads: u64,
    pub gdn_v_heads: u64,
    pub gdn_conv: u64,
    /// v heads x state: the value width
    pub gdn_value: u64,
    /// hyper-connections
    pub hc: u64,
    pub hc_rank: u64,
    /// PLE
    pub ple_layers: Vec<u64>,
    pub ple_dim: u64,
    pub ple_ngram: u64,
    pub ple_heads_per_ngram: u64,
    pub ple_conv: u64,
    pub ple_eos: u32,
    pub ple_mult: Vec<u64>,
    pub ple_offsets: Vec<u64>,
    pub ple_vocab: Vec<u64>,
    pub ple_rows: u64,
    /// true for the QSA layers
    pub qsa: Vec<bool>,
}

impl Geometry {
    pub fn is_qsa(&self, layer: u64) -> bool {
        self.qsa.get(layer as usize).copied().unwrap_or(false)
    }
    pub fn has_ple(&self, layer: u64) -> bool {
        self.ple_layers.contains(&layer)
    }
    /// the GDN conv's channels: q and k (k heads x state each), then v
    pub fn gdn_channels(&self) -> u64 {
        2 * self.gdn_k_heads * self.gdn_state + self.gdn_value
    }
    /// PLE rows a token reads (n-gram orders 2..=ngram, `heads_per_ngram` each)
    pub fn ple_heads(&self) -> u64 {
        (self.ple_ngram - 1) * self.ple_heads_per_ngram
    }
}

/// A tensor's role. Per layer unless global.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    TokenEmbd,
    Output,
    OutputHcNorm,
    OutputHcDown,
    OutputHcUp,
    PleTable,
    // every layer: the two hyper-connection reads / writes
    HcAttnNorm,
    HcAttnDown,
    HcAttnUp,
    HcAttnInject,
    HcFfnNorm,
    HcFfnDown,
    HcFfnUp,
    HcFfnInject,
    // Gated DeltaNet
    GdnQkv,
    GdnZ,
    GdnAlpha,
    GdnBeta,
    GdnConv,
    GdnA,
    GdnDtBias,
    GdnNorm,
    GdnOut,
    // QSA
    QsaQ,
    QsaK,
    QsaV,
    QsaOut,
    QsaQNorm,
    QsaKNorm,
    IdxQ,
    IdxK,
    IdxQNorm,
    IdxKNorm,
    // MoE
    Router,
    ShGateInp,
    ExpGate,
    ExpUp,
    ExpDown,
    ShGate,
    ShUp,
    ShDown,
    // PLE (its layers)
    PleKey,
    PleValue,
    PleNormKey,
    PleNormQuery,
    PleNormConv,
    PleConv,
}

/// What a role belongs to, for the memory plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// embedding and the head
    Edges,
    /// attention of either kind, hyper-connections
    Attention,
    /// the router and the shared expert: every token, always resident
    Shared,
    /// the routed experts: 10 of 512 a token
    Experts,
    /// the PLE block's projections
    Ple,
    /// the hashed per-layer table (host memory: 16 rows a token)
    PleTable,
}

impl Role {
    pub fn group(self) -> Group {
        use Role::*;
        match self {
            TokenEmbd | Output | OutputHcNorm | OutputHcDown | OutputHcUp => Group::Edges,
            PleTable => Group::PleTable,
            Router | ShGateInp | ShGate | ShUp | ShDown => Group::Shared,
            ExpGate | ExpUp | ExpDown => Group::Experts,
            PleKey | PleValue | PleNormKey | PleNormQuery | PleNormConv | PleConv => Group::Ple,
            _ => Group::Attention,
        }
    }

    /// The tensor's name suffix in a file (after `blk.N.`; the whole name for a global role).
    pub fn name(self) -> &'static str {
        use Role::*;
        match self {
            TokenEmbd => "token_embd.weight",
            Output => "output.weight",
            OutputHcNorm => "output_hc_norm.weight",
            OutputHcDown => "output_hc_down.weight",
            OutputHcUp => "output_hc_up.weight",
            PleTable => "per_layer_token_embd.weight",
            HcAttnNorm => "hc_attn_norm.weight",
            HcAttnDown => "hc_attn_down.weight",
            HcAttnUp => "hc_attn_up.weight",
            HcAttnInject => "hc_attn_inject.weight",
            HcFfnNorm => "hc_ffn_norm.weight",
            HcFfnDown => "hc_ffn_down.weight",
            HcFfnUp => "hc_ffn_up.weight",
            HcFfnInject => "hc_ffn_inject.weight",
            GdnQkv => "attn_qkv.weight",
            GdnZ => "attn_gate.weight",
            GdnAlpha => "ssm_alpha.weight",
            GdnBeta => "ssm_beta.weight",
            GdnConv => "ssm_conv1d.weight",
            GdnA => "ssm_a",
            GdnDtBias => "ssm_dt.bias",
            GdnNorm => "ssm_norm.weight",
            GdnOut => "ssm_out.weight",
            QsaQ => "attn_q.weight",
            QsaK => "attn_k.weight",
            QsaV => "attn_v.weight",
            QsaOut => "attn_output.weight",
            QsaQNorm => "attn_q_norm.weight",
            QsaKNorm => "attn_k_norm.weight",
            IdxQ => "indexer.q_proj.weight",
            IdxK => "indexer.k_proj.weight",
            IdxQNorm => "indexer.q_norm.weight",
            IdxKNorm => "indexer.k_norm.weight",
            Router => "ffn_gate_inp.weight",
            ShGateInp => "ffn_gate_inp_shexp.weight",
            ExpGate => "ffn_gate_exps.weight",
            ExpUp => "ffn_up_exps.weight",
            ExpDown => "ffn_down_exps.weight",
            ShGate => "ffn_gate_shexp.weight",
            ShUp => "ffn_up_shexp.weight",
            ShDown => "ffn_down_shexp.weight",
            PleKey => "ple_key.weight",
            PleValue => "ple_value.weight",
            PleNormKey => "ple_norm_key.weight",
            PleNormQuery => "ple_norm_query.weight",
            PleNormConv => "ple_norm_conv.weight",
            PleConv => "ple_conv1d.weight",
        }
    }

    pub fn global(self) -> bool {
        use Role::*;
        matches!(self, TokenEmbd | Output | OutputHcNorm | OutputHcDown | OutputHcUp | PleTable)
    }
}

pub const GLOBAL: [Role; 6] = [Role::TokenEmbd, Role::Output, Role::OutputHcNorm, Role::OutputHcDown, Role::OutputHcUp, Role::PleTable];
const HC: [Role; 8] = [Role::HcAttnNorm, Role::HcAttnDown, Role::HcAttnUp, Role::HcAttnInject, Role::HcFfnNorm, Role::HcFfnDown, Role::HcFfnUp, Role::HcFfnInject];
const GDN: [Role; 9] = [Role::GdnQkv, Role::GdnZ, Role::GdnAlpha, Role::GdnBeta, Role::GdnConv, Role::GdnA, Role::GdnDtBias, Role::GdnNorm, Role::GdnOut];
const QSA: [Role; 10] = [Role::QsaQ, Role::QsaK, Role::QsaV, Role::QsaOut, Role::QsaQNorm, Role::QsaKNorm, Role::IdxQ, Role::IdxK, Role::IdxQNorm, Role::IdxKNorm];
const MOE: [Role; 8] = [Role::Router, Role::ShGateInp, Role::ExpGate, Role::ExpUp, Role::ExpDown, Role::ShGate, Role::ShUp, Role::ShDown];
const PLE: [Role; 6] = [Role::PleKey, Role::PleValue, Role::PleNormKey, Role::PleNormQuery, Role::PleNormConv, Role::PleConv];

/// A model file, resolved: its geometry and every tensor by (layer, role).
pub struct Model<'g> {
    pub file: &'g Gguf,
    pub g: Geometry,
}

fn meta_u(f: &Gguf, key: &str) -> Result<u64> {
    f.arch_meta(key).and_then(Value::as_u64).ok_or_else(|| Error(format!("metadata {key} is missing")))
}
fn meta_f(f: &Gguf, key: &str) -> Result<f64> {
    f.arch_meta(key).and_then(Value::as_f64).ok_or_else(|| Error(format!("metadata {key} is missing")))
}
fn meta_list(f: &Gguf, key: &str) -> Result<Vec<u64>> {
    f.arch_meta(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .ok_or_else(|| Error(format!("metadata {key} is missing")))
}

impl<'g> Model<'g> {
    pub fn open(file: &'g Gguf) -> Result<Model<'g>> {
        let n_layer = meta_u(file, "block_count")?;
        // the layer pattern: compress_ratios (non-zero = a QSA layer), else full_attention_interval
        let qsa: Vec<bool> = match meta_list(file, "attention.compress_ratios") {
            Ok(r) if r.len() as u64 == n_layer => r.iter().map(|c| *c != 0).collect(),
            _ => {
                let iv = meta_u(file, "full_attention_interval")?;
                (0..n_layer).map(|l| l % iv == iv - 1).collect()
            }
        };
        let idx_block = meta_list(file, "attention.compress_ratios").ok().and_then(|r| r.into_iter().find(|c| *c != 0)).unwrap_or(4);
        let shape = |name: &str| file.tensor(name).map(|t| t.shape.clone()).ok_or_else(|| Error(format!("{name} is missing")));
        let gdn_state = meta_u(file, "ssm.state_size")?;
        let ple_table = shape(Role::PleTable.name())?; // [rows, dim]
        let g = Geometry {
            n_embd: meta_u(file, "embedding_length")?,
            n_vocab: shape(Role::TokenEmbd.name())?[0],
            n_layer,
            rms_eps: meta_f(file, "attention.layer_norm_rms_epsilon")?,
            n_expert: meta_u(file, "expert_count")?,
            n_expert_used: meta_u(file, "expert_used_count")?,
            ffn_expert: meta_u(file, "expert_feed_forward_length")?,
            ffn_shared: meta_u(file, "expert_shared_feed_forward_length")?,
            n_head: meta_u(file, "attention.head_count")?,
            n_head_kv: meta_u(file, "attention.head_count_kv")?,
            head_dim: meta_u(file, "attention.key_length")?,
            n_rot: meta_u(file, "rope.dimension_count")?,
            rope_base: meta_f(file, "rope.freq_base")?,
            idx_heads: meta_u(file, "attention.indexer.head_count")?,
            idx_dim: meta_u(file, "attention.indexer.key_length")?,
            idx_top_k: meta_u(file, "attention.indexer.top_k")?,
            idx_block,
            gdn_state,
            gdn_k_heads: meta_u(file, "ssm.group_count")?,
            gdn_v_heads: meta_u(file, "ssm.time_step_rank")?,
            gdn_conv: meta_u(file, "ssm.conv_kernel")?,
            gdn_value: meta_u(file, "ssm.inner_size")?,
            hc: meta_u(file, "hyper_connection.count")?,
            hc_rank: meta_u(file, "hyper_connection.low_rank")?,
            ple_layers: meta_list(file, "ple.layers")?,
            ple_dim: meta_u(file, "embedding_length_per_layer_input")?,
            ple_ngram: meta_u(file, "ple.ngram_size")?,
            ple_heads_per_ngram: meta_u(file, "ple.heads_per_ngram")?,
            ple_conv: meta_u(file, "ple.conv_kernel")?,
            ple_eos: meta_u(file, "ple.eos_token_id")? as u32,
            ple_mult: meta_list(file, "ple.layer_multipliers")?,
            ple_offsets: meta_list(file, "ple.head_offsets")?,
            ple_vocab: meta_list(file, "ple.head_vocab_sizes")?,
            ple_rows: ple_table[0],
            qsa,
        };
        Ok(Model { file, g })
    }

    /// A tensor by role (`layer` ignored for a global role).
    pub fn tensor(&self, layer: u64, r: Role) -> Option<&'g Tensor> {
        if r.global() {
            self.file.tensor(r.name())
        } else {
            self.file.tensor(&format!("blk.{layer}.{}", r.name()))
        }
    }

    /// A tensor that `check` has vouched for.
    pub fn t(&self, layer: u64, r: Role) -> &'g Tensor {
        self.tensor(layer, r).unwrap_or_else(|| panic!("blk.{layer}.{}: checked at load", r.name()))
    }

    /// The roles layer `l` has.
    pub fn roles(&self, l: u64) -> Vec<Role> {
        let mut v = HC.to_vec();
        v.extend(if self.g.is_qsa(l) { QSA.as_slice() } else { GDN.as_slice() });
        v.extend(MOE);
        if self.g.has_ple(l) {
            v.extend(PLE);
        }
        v
    }

    /// The shape a role must have (row-major, outermost first).
    pub fn shape(&self, r: Role) -> Vec<u64> {
        use Role::*;
        let g = &self.g;
        let (d, hd) = (g.n_embd, g.hc * g.n_embd);
        let ch = g.gdn_channels();
        match r {
            TokenEmbd | Output => vec![g.n_vocab, d],
            OutputHcNorm | HcAttnNorm | HcFfnNorm => vec![hd],
            OutputHcDown | HcAttnDown | HcFfnDown => vec![g.hc_rank, hd],
            OutputHcUp | HcAttnUp | HcFfnUp => vec![hd, g.hc_rank],
            HcAttnInject | HcFfnInject => vec![g.hc, hd],
            PleTable => vec![g.ple_rows, g.ple_dim],
            GdnQkv => vec![ch, d],
            GdnZ => vec![g.gdn_value, d],
            GdnAlpha | GdnBeta => vec![g.gdn_v_heads, d],
            GdnConv => vec![ch, g.gdn_conv],
            GdnA | GdnDtBias => vec![g.gdn_v_heads],
            GdnNorm => vec![g.gdn_state],
            GdnOut => vec![d, g.gdn_value],
            QsaQ => vec![2 * g.n_head * g.head_dim, d],
            QsaK | QsaV => vec![g.n_head_kv * g.head_dim, d],
            QsaOut => vec![d, g.n_head * g.head_dim],
            QsaQNorm | QsaKNorm => vec![g.head_dim],
            IdxQ => vec![g.idx_heads * g.idx_dim, d],
            IdxK => vec![g.idx_dim, d],
            IdxQNorm | IdxKNorm => vec![g.idx_dim],
            Router => vec![g.n_expert, d],
            ShGateInp => vec![d],
            ExpGate | ExpUp => vec![g.n_expert, g.ffn_expert, d],
            ExpDown => vec![g.n_expert, d, g.ffn_expert],
            ShGate | ShUp => vec![g.ffn_shared, d],
            ShDown => vec![d, g.ffn_shared],
            // both read a token's PLE rows (heads x dim): the key against all streams, the value for one
            PleKey => vec![hd, g.ple_heads() * g.ple_dim],
            PleValue => vec![d, g.ple_heads() * g.ple_dim],
            PleNormKey | PleNormQuery | PleNormConv => vec![hd],
            PleConv => vec![hd, g.ple_conv],
        }
    }

    /// Every layer's every role: present and of the right shape; no tensor in the file left over; the metadata
    /// consistent. All problems, not the first.
    pub fn check(&self) -> Vec<String> {
        let g = &self.g;
        let mut errs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for r in GLOBAL {
            self.check_one(0, r, &mut errs, &mut seen);
        }
        for l in 0..g.n_layer {
            for r in self.roles(l) {
                self.check_one(l, r, &mut errs, &mut seen);
            }
        }
        for t in &self.file.tensors {
            if !seen.contains(t.name.as_str()) {
                errs.push(format!("{}: in the file, not a role the runtime knows ({:?} {})", t.name, t.shape, t.ty.name()));
            }
        }
        let heads = g.ple_heads() as usize;
        if g.ple_offsets.len() != heads || g.ple_vocab.len() != heads {
            errs.push(format!("ple: {} offsets and {} vocab sizes, {heads} heads", g.ple_offsets.len(), g.ple_vocab.len()));
        } else if g.ple_offsets.iter().zip(&g.ple_vocab).any(|(o, v)| o + v > g.ple_rows) {
            errs.push(format!("ple: a head's rows end past the table's {}", g.ple_rows));
        }
        if g.ple_mult.len() as u64 != g.ple_ngram {
            errs.push(format!("ple: {} multipliers for {}-grams", g.ple_mult.len(), g.ple_ngram));
        }
        if g.gdn_v_heads * g.gdn_state != g.gdn_value || !g.gdn_v_heads.is_multiple_of(g.gdn_k_heads) {
            errs.push(format!("gdn: {} v heads of {} for a value width {} and {} k heads", g.gdn_v_heads, g.gdn_state, g.gdn_value, g.gdn_k_heads));
        }
        errs
    }

    fn check_one<'a>(&'a self, l: u64, r: Role, errs: &mut Vec<String>, seen: &mut std::collections::HashSet<&'a str>) {
        match self.tensor(l, r) {
            None => errs.push(format!("layer {l}: {:?} ({}) is missing", r, r.name())),
            Some(t) => {
                seen.insert(t.name.as_str());
                let want = self.shape(r);
                if t.shape != want {
                    errs.push(format!("{}: shape {:?}, the runtime needs {:?}", t.name, t.shape, want));
                }
            }
        }
    }

    /// Bytes by group and type, for the memory plan and `info`.
    pub fn bytes(&self) -> std::collections::BTreeMap<(Group, GType), (u64, u64)> {
        let mut m = std::collections::BTreeMap::new();
        let mut add = |r: Role, t: &Tensor| {
            let e = m.entry((r.group(), t.ty)).or_insert((0u64, 0u64));
            e.0 += t.bytes;
            e.1 += 1;
        };
        for r in GLOBAL {
            if let Some(t) = self.tensor(0, r) {
                add(r, t);
            }
        }
        for l in 0..self.g.n_layer {
            for r in self.roles(l) {
                if let Some(t) = self.tensor(l, r) {
                    add(r, t);
                }
            }
        }
        m
    }

    /// One routed expert's bytes in layer `l` (gate + up + down rows).
    pub fn expert_bytes(&self, l: u64) -> u64 {
        [Role::ExpGate, Role::ExpUp, Role::ExpDown].iter().filter_map(|r| self.tensor(l, *r)).map(|t| t.bytes / self.g.n_expert).sum()
    }
}
