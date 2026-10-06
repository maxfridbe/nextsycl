//! GLM-5.3-Flash (`glm5-next`): 45 trunk layers and an MTP block, hidden 4096.
//!
//! ```text
//!   every layer:   hyper-connections around both halves (4 streams; hc_*_fn [24, 4 x hidden]: 4 pre + 4 post
//!                  + a 4x4 residual mix, Sinkhorn-normalized), then
//!   attention:     KDA (Kimi delta attention, 3 of every 4 layers): q/k/v [64 x 128] with short convolutions,
//!                  a per-channel decay from a low-rank gate (f_a -> f_b), beta per head, an output gate (g_a ->
//!                  g_b), a head norm; or
//!                  MLA (every 4th: layers 3, 7, ..., 43): q through a 1536 LoRA, keys/values through a shared
//!                  512 latent (k_b / v_b per head, 64 heads of 256), no rotary; a DSA lightning indexer (32
//!                  heads of 128, top-2048 keys, a 4-way pooled key compressor) picks what each query reads
//!   feed-forward:  dense (layers 0-2, 12288) or MoE (288 experts of 2048, top-8 by a sigmoid gate with a bias,
//!                  normalized, x2.5, plus one shared expert; SwiGLU clamped at 10)
//!   MTP:           one extra block (the ds4 file's layer 45) with nextn.{eh_proj, enorm, hnorm, shared_head_norm}
//! ```
//!
//! Two converters' files exist: llama.cpp's (PR 27773; `ssm_*` for KDA, `indexer_compressor_*`, per-layer
//! `attention.head_count_kv` for the layer pattern) and antirez ds4's (`kda_*`, `indexer.pool_*`, `layer_types`,
//! MTP kept). `Scheme` maps the runtime's roles onto either.

use ns_gguf::{GType, Gguf, Tensor, Value};

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// llama.cpp's names (de25343 / PR 27773)
    LlamaCpp,
    /// antirez ds4's names
    Ds4,
}

/// The geometry, from metadata and the tensors' shapes.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub n_embd: u64,
    pub n_vocab: u64,
    /// trunk layers (not the MTP block)
    pub n_layer: u64,
    /// MTP blocks after the trunk (0 or 1)
    pub n_mtp: u64,
    pub n_dense: u64,
    pub ffn_dense: u64,
    pub n_expert: u64,
    pub n_expert_used: u64,
    pub n_expert_shared: u64,
    pub ffn_expert: u64,
    pub expert_scale: f64,
    pub expert_norm: bool,
    pub swiglu_limit: f64,
    pub rms_eps: f64,
    /// MLA
    pub n_head: u64,
    pub q_lora: u64,
    pub kv_lora: u64,
    pub head_dim: u64,
    /// the lightning indexer
    pub idx_heads: u64,
    pub idx_dim: u64,
    pub idx_top_k: u64,
    pub idx_pool: u64,
    /// KDA
    pub kda_heads: u64,
    pub kda_dim: u64,
    pub kda_conv: u64,
    pub kda_gate_low: f64,
    pub kda_rank: u64,
    /// hyper-connections
    pub hc: u64,
    pub hc_iters: u64,
    pub hc_eps: f64,
    /// true for the MLA layers, per trunk layer
    pub mla: Vec<bool>,
}

impl Geometry {
    pub fn is_mla(&self, layer: u64) -> bool {
        self.mla.get(layer as usize).copied().unwrap_or(false)
    }
    pub fn is_moe(&self, layer: u64) -> bool {
        layer >= self.n_dense
    }
}

/// A tensor's role. `L` = in a trunk layer, `M` = in the MTP block, the rest global.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    TokenEmbd,
    OutputNorm,
    Output,
    // every layer
    AttnNorm,
    FfnNorm,
    HcAttnFn,
    HcAttnBase,
    HcAttnScale,
    HcFfnFn,
    HcFfnBase,
    HcFfnScale,
    // KDA
    KdaQ,
    KdaK,
    KdaV,
    KdaQConv,
    KdaKConv,
    KdaVConv,
    KdaFA,
    KdaFB,
    KdaDtBias,
    /// llama.cpp stores `ssm_a`, ds4 `kda_a_log`: the same per-head decay rate in two forms (see `a_form`)
    KdaA,
    KdaBeta,
    KdaGA,
    KdaGB,
    KdaONorm,
    KdaOut,
    // MLA
    MlaQA,
    MlaQANorm,
    MlaQB,
    MlaKvA,
    MlaKvANorm,
    MlaKB,
    MlaVB,
    MlaOut,
    // the indexer
    IdxQB,
    IdxK,
    IdxKNorm,
    IdxKNormBias,
    IdxProj,
    IdxPoolApe,
    IdxPoolGate,
    // dense feed-forward
    FfnGate,
    FfnUp,
    FfnDown,
    // MoE
    Router,
    RouterBias,
    ExpGate,
    ExpUp,
    ExpDown,
    ShGate,
    ShUp,
    ShDown,
    // MTP
    MtpEhProj,
    MtpENorm,
    MtpHNorm,
    MtpHeadNorm,
}

/// What a role belongs to, for the memory plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// embedding and the head
    Edges,
    /// attention of either kind, norms, hyper-connections
    Attention,
    Dense,
    /// the router and the shared expert: on every token, always resident
    Shared,
    /// the routed experts: 8 of 288 per token, what does not fit streams
    Experts,
    Mtp,
}

impl Role {
    pub fn group(self) -> Group {
        use Role::*;
        match self {
            TokenEmbd | OutputNorm | Output => Group::Edges,
            FfnGate | FfnUp | FfnDown => Group::Dense,
            Router | RouterBias | ShGate | ShUp | ShDown => Group::Shared,
            ExpGate | ExpUp | ExpDown => Group::Experts,
            MtpEhProj | MtpENorm | MtpHNorm | MtpHeadNorm => Group::Mtp,
            _ => Group::Attention,
        }
    }

    /// The tensor's name suffix in a file (after `blk.N.`; the whole name for a global role).
    pub fn name(self, s: Scheme) -> &'static str {
        use Role::*;
        let ll = s == Scheme::LlamaCpp;
        match self {
            TokenEmbd => "token_embd.weight",
            OutputNorm => "output_norm.weight",
            Output => "output.weight",
            AttnNorm => "attn_norm.weight",
            FfnNorm => "ffn_norm.weight",
            HcAttnFn => "hc_attn_fn.weight",
            HcAttnBase => "hc_attn_base.weight",
            HcAttnScale => "hc_attn_scale.weight",
            HcFfnFn => "hc_ffn_fn.weight",
            HcFfnBase => "hc_ffn_base.weight",
            HcFfnScale => "hc_ffn_scale.weight",
            KdaQ => if ll { "attn_q.weight" } else { "kda_q.weight" },
            KdaK => if ll { "attn_k.weight" } else { "kda_k.weight" },
            KdaV => if ll { "attn_v.weight" } else { "kda_v.weight" },
            KdaQConv => if ll { "ssm_conv1d_q.weight" } else { "kda_q_conv.weight" },
            KdaKConv => if ll { "ssm_conv1d_k.weight" } else { "kda_k_conv.weight" },
            KdaVConv => if ll { "ssm_conv1d_v.weight" } else { "kda_v_conv.weight" },
            KdaFA => if ll { "ssm_f_a.weight" } else { "kda_f_a.weight" },
            KdaFB => if ll { "ssm_f_b.weight" } else { "kda_f_b.weight" },
            KdaDtBias => if ll { "ssm_dt.bias" } else { "kda_dt_bias.weight" },
            KdaA => if ll { "ssm_a" } else { "kda_a_log.weight" },
            KdaBeta => if ll { "ssm_beta.weight" } else { "kda_beta.weight" },
            KdaGA => if ll { "ssm_g_a.weight" } else { "kda_g_a.weight" },
            KdaGB => if ll { "ssm_g_b.weight" } else { "kda_g_b.weight" },
            KdaONorm => if ll { "ssm_norm.weight" } else { "kda_o_norm.weight" },
            KdaOut => if ll { "attn_output.weight" } else { "kda_output.weight" },
            MlaQA => "attn_q_a.weight",
            MlaQANorm => "attn_q_a_norm.weight",
            MlaQB => "attn_q_b.weight",
            MlaKvA => "attn_kv_a_mqa.weight",
            MlaKvANorm => "attn_kv_a_norm.weight",
            MlaKB => "attn_k_b.weight",
            MlaVB => "attn_v_b.weight",
            MlaOut => "attn_output.weight",
            IdxQB => "indexer.attn_q_b.weight",
            IdxK => "indexer.attn_k.weight",
            IdxKNorm => "indexer.k_norm.weight",
            IdxKNormBias => "indexer.k_norm.bias",
            IdxProj => "indexer.proj.weight",
            IdxPoolApe => if ll { "indexer_compressor_ape.weight" } else { "indexer.pool_ape.weight" },
            IdxPoolGate => if ll { "indexer_compressor_gate.weight" } else { "indexer.pool_gate.weight" },
            FfnGate => "ffn_gate.weight",
            FfnUp => "ffn_up.weight",
            FfnDown => "ffn_down.weight",
            Router => "ffn_gate_inp.weight",
            RouterBias => "exp_probs_b.bias",
            ExpGate => "ffn_gate_exps.weight",
            ExpUp => "ffn_up_exps.weight",
            ExpDown => "ffn_down_exps.weight",
            ShGate => "ffn_gate_shexp.weight",
            ShUp => "ffn_up_shexp.weight",
            ShDown => "ffn_down_shexp.weight",
            MtpEhProj => "nextn.eh_proj.weight",
            MtpENorm => "nextn.enorm.weight",
            MtpHNorm => "nextn.hnorm.weight",
            MtpHeadNorm => "nextn.shared_head_norm.weight",
        }
    }

    fn global(self) -> bool {
        matches!(self, Role::TokenEmbd | Role::OutputNorm | Role::Output)
    }
}

const COMMON: [Role; 8] = [Role::AttnNorm, Role::FfnNorm, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale];
const KDA: [Role; 15] = [Role::KdaQ, Role::KdaK, Role::KdaV, Role::KdaQConv, Role::KdaKConv, Role::KdaVConv, Role::KdaFA, Role::KdaFB, Role::KdaDtBias, Role::KdaA,
                         Role::KdaBeta, Role::KdaGA, Role::KdaGB, Role::KdaONorm, Role::KdaOut];
const MLA: [Role; 15] = [Role::MlaQA, Role::MlaQANorm, Role::MlaQB, Role::MlaKvA, Role::MlaKvANorm, Role::MlaKB, Role::MlaVB, Role::MlaOut, Role::IdxQB, Role::IdxK,
                         Role::IdxKNorm, Role::IdxKNormBias, Role::IdxProj, Role::IdxPoolApe, Role::IdxPoolGate];
const DENSE: [Role; 3] = [Role::FfnGate, Role::FfnUp, Role::FfnDown];
const MOE: [Role; 8] = [Role::Router, Role::RouterBias, Role::ExpGate, Role::ExpUp, Role::ExpDown, Role::ShGate, Role::ShUp, Role::ShDown];
const MTP: [Role; 4] = [Role::MtpEhProj, Role::MtpENorm, Role::MtpHNorm, Role::MtpHeadNorm];

/// A model file, resolved: its scheme, geometry and every tensor by (layer, role).
pub struct Model<'g> {
    pub file: &'g Gguf,
    pub scheme: Scheme,
    pub g: Geometry,
}

fn meta_u(f: &Gguf, keys: &[&str]) -> Result<u64> {
    keys.iter().find_map(|k| f.arch_meta(k).and_then(Value::as_u64)).ok_or_else(|| Error(format!("metadata {} is missing", keys[0])))
}
fn meta_f(f: &Gguf, keys: &[&str], default: Option<f64>) -> Result<f64> {
    keys.iter()
        .find_map(|k| f.arch_meta(k).and_then(|v| v.as_f64().or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_f64))))
        .or(default)
        .ok_or_else(|| Error(format!("metadata {} is missing", keys[0])))
}

impl<'g> Model<'g> {
    pub fn open(file: &'g Gguf) -> Result<Model<'g>> {
        let scheme = if file.tensor("blk.0.kda_q.weight").is_some() { Scheme::Ds4 } else { Scheme::LlamaCpp };
        let trunk = match scheme {
            Scheme::Ds4 => meta_u(file, &["trunk_block_count"])?,
            Scheme::LlamaCpp => meta_u(file, &["block_count"])?,
        };
        let blocks = meta_u(file, &["block_count"])?;
        // the layer pattern: ds4 `layer_types` (1 = MLA), llama.cpp `attention.head_count_kv` (1 = full attention)
        let pattern = file.arch_meta("layer_types").or_else(|| file.arch_meta("attention.head_count_kv")).and_then(Value::as_array)
            .ok_or_else(|| Error("neither layer_types nor attention.head_count_kv gives the layer pattern".into()))?;
        let mla: Vec<bool> = (0..trunk as usize).map(|i| pattern.get(i % pattern.len()).and_then(Value::as_u64).unwrap_or(0) != 0).collect();
        // shapes the metadata names differently in the two schemes are taken from the tensors
        let shape = |name: &str| file.tensor(name).map(|t| t.shape.clone()).ok_or_else(|| Error(format!("{name} is missing")));
        let mla_layer = mla.iter().position(|m| *m).ok_or("no MLA layer in the pattern")? as u64;
        let kb = shape(&format!("blk.{mla_layer}.{}", Role::MlaKB.name(scheme)))?; // [heads, kv_lora, head_dim]
        let fa = shape(&format!("blk.0.{}", Role::KdaFA.name(scheme)))?; // [rank, hidden]
        let g = Geometry {
            n_embd: meta_u(file, &["embedding_length"])?,
            n_vocab: meta_u(file, &["vocab_size"]).or_else(|_| shape(Role::TokenEmbd.name(scheme)).map(|s| s[0]))?,
            n_layer: trunk,
            n_mtp: blocks - trunk.min(blocks),
            n_dense: meta_u(file, &["leading_dense_block_count"])?,
            ffn_dense: meta_u(file, &["feed_forward_length"])?,
            n_expert: meta_u(file, &["expert_count"])?,
            n_expert_used: meta_u(file, &["expert_used_count"])?,
            n_expert_shared: meta_u(file, &["expert_shared_count"])?,
            ffn_expert: meta_u(file, &["expert_feed_forward_length"])?,
            expert_scale: meta_f(file, &["expert_weights_scale"], None)?,
            expert_norm: meta_u(file, &["expert_weights_norm"]).unwrap_or(1) != 0,
            swiglu_limit: meta_f(file, &["swiglu_limit", "swiglu_clamp_exp"], None)?,
            rms_eps: meta_f(file, &["attention.layer_norm_rms_epsilon"], None)?,
            n_head: meta_u(file, &["attention.head_count"])?,
            q_lora: meta_u(file, &["attention.q_lora_rank"])?,
            kv_lora: meta_u(file, &["attention.kv_lora_rank"])?,
            head_dim: kb[2],
            idx_heads: meta_u(file, &["attention.indexer.head_count"])?,
            idx_dim: meta_u(file, &["attention.indexer.key_length"])?,
            idx_top_k: meta_u(file, &["attention.indexer.top_k"])?,
            idx_pool: meta_u(file, &["attention.indexer.pool_size", "attention.indexer.kpool"])?,
            kda_heads: meta_u(file, &["linear_attention.head_count"]).or_else(|_| shape(&format!("blk.0.{}", Role::KdaA.name(scheme))).map(|s| s[0]))?,
            kda_dim: meta_u(file, &["linear_attention.head_dimension", "kda.head_dim"])?,
            kda_conv: meta_u(file, &["linear_attention.conv_kernel", "ssm.conv_kernel"])?,
            kda_gate_low: meta_f(file, &["linear_attention.gate_lower_bound", "kda.gate_lower_bound"], None)?,
            kda_rank: fa[0],
            hc: meta_u(file, &["hyper_connection.count"])?,
            hc_iters: meta_u(file, &["hyper_connection.sinkhorn_iterations"])?,
            hc_eps: meta_f(file, &["hyper_connection.epsilon"], None)?,
            mla,
        };
        Ok(Model { file, scheme, g })
    }

    /// A tensor by role: `layer` is a trunk layer, or `n_layer + i` for MTP block i; ignored for global roles.
    pub fn tensor(&self, layer: u64, r: Role) -> Option<&'g Tensor> {
        if r.global() {
            self.file.tensor(r.name(self.scheme))
        } else {
            self.file.tensor(&format!("blk.{layer}.{}", r.name(self.scheme)))
        }
    }

    /// The roles block `layer` has (a trunk layer, or an MTP block - which is an MLA MoE layer plus its own four).
    pub fn roles(&self, layer: u64) -> Vec<Role> {
        let g = &self.g;
        let mtp = layer >= g.n_layer;
        let mut v: Vec<Role> = if mtp { COMMON[..2].to_vec() } else { COMMON.to_vec() };
        if mtp || g.is_mla(layer) {
            v.extend(MLA);
        } else {
            v.extend(KDA);
        }
        if mtp || g.is_moe(layer) {
            v.extend(MOE);
        } else {
            v.extend(DENSE);
        }
        if mtp {
            v.extend(MTP);
        }
        v
    }

    /// The shape a role must have (row-major, outermost first).
    pub fn shape(&self, r: Role) -> Vec<u64> {
        use Role::*;
        let g = &self.g;
        let (d, hc) = (g.n_embd, g.hc);
        let kda_w = g.kda_heads * g.kda_dim;
        let mla_w = g.n_head * g.head_dim;
        match r {
            TokenEmbd | Output => vec![g.n_vocab, d],
            OutputNorm | AttnNorm | FfnNorm | MtpENorm | MtpHNorm | MtpHeadNorm => vec![d],
            HcAttnFn | HcFfnFn => vec![hc * 2 + hc * hc, hc * d],
            HcAttnBase | HcFfnBase => vec![hc * 2 + hc * hc],
            HcAttnScale | HcFfnScale => vec![3],
            KdaQ | KdaK | KdaV => vec![kda_w, d],
            KdaQConv | KdaKConv | KdaVConv => vec![kda_w, 1, g.kda_conv],
            KdaFA | KdaGA => vec![g.kda_rank, d],
            KdaFB | KdaGB => vec![kda_w, g.kda_rank],
            KdaDtBias => vec![kda_w],
            KdaA => vec![g.kda_heads],
            KdaBeta => vec![g.kda_heads, d],
            KdaONorm => vec![g.kda_dim],
            KdaOut => vec![d, kda_w],
            MlaQA => vec![g.q_lora, d],
            MlaQANorm => vec![g.q_lora],
            MlaQB => vec![mla_w, g.q_lora],
            MlaKvA => vec![g.kv_lora, d],
            MlaKvANorm => vec![g.kv_lora],
            MlaKB => vec![g.n_head, g.kv_lora, g.head_dim],
            MlaVB => vec![g.n_head, g.head_dim, g.kv_lora],
            MlaOut => vec![d, mla_w],
            IdxQB => vec![g.idx_heads * g.idx_dim, g.q_lora],
            IdxK => vec![g.idx_dim, d],
            IdxKNorm | IdxKNormBias => vec![g.idx_dim],
            IdxProj => vec![g.idx_heads, d],
            IdxPoolApe => vec![g.idx_pool, g.idx_dim],
            IdxPoolGate => vec![g.idx_dim, d],
            FfnGate | FfnUp => vec![g.ffn_dense, d],
            FfnDown => vec![d, g.ffn_dense],
            Router => vec![g.n_expert, d],
            RouterBias => vec![g.n_expert],
            ExpGate | ExpUp => vec![g.n_expert, g.ffn_expert, d],
            ExpDown => vec![g.n_expert, d, g.ffn_expert],
            ShGate | ShUp => vec![g.ffn_expert * g.n_expert_shared, d],
            ShDown => vec![d, g.ffn_expert * g.n_expert_shared],
            MtpEhProj => vec![d, 2 * d],
        }
    }

    /// Every block's every role: present and of the right shape; no tensor in the file left over. All problems,
    /// not the first.
    pub fn check(&self) -> Vec<String> {
        let mut errs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for r in [Role::TokenEmbd, Role::OutputNorm, Role::Output] {
            self.check_one(0, r, &mut errs, &mut seen);
        }
        for l in 0..self.g.n_layer + self.g.n_mtp {
            for r in self.roles(l) {
                self.check_one(l, r, &mut errs, &mut seen);
            }
        }
        for t in &self.file.tensors {
            if !seen.contains(t.name.as_str()) {
                errs.push(format!("{}: in the file, not a role the runtime knows ({:?} {})", t.name, t.shape, t.ty.name()));
            }
        }
        errs
    }

    fn check_one<'a>(&'a self, l: u64, r: Role, errs: &mut Vec<String>, seen: &mut std::collections::HashSet<&'a str>) {
        match self.tensor(l, r) {
            None => errs.push(format!("block {l}: {:?} ({}) is missing", r, r.name(self.scheme))),
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
        for r in [Role::TokenEmbd, Role::OutputNorm, Role::Output] {
            if let Some(t) = self.tensor(0, r) {
                add(r, t);
            }
        }
        for l in 0..self.g.n_layer + self.g.n_mtp {
            for r in self.roles(l) {
                if let Some(t) = self.tensor(l, r) {
                    // the MTP block's own attention / MoE count as MTP, not as trunk
                    add(if l >= self.g.n_layer { Role::MtpEhProj } else { r }, t);
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
