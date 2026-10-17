//! DeepSeek-V4.1-Flash (`deepseek4` files of the variant `v41_flash`): 40 layers, hidden 5120, the residual as 4 mHC
//! streams; YoungAi's VQ GGUF (routed experts vector-quantized, the rest q4_K / fp8_32x32 / BF16 / F32).
//!
//! ```text
//!   every block:   hyper-connections around both halves (V4's mixes and Sinkhorn), but each half's input collapsed by
//!                  the mix the half BEFORE it computed: the attention by the previous layer's FFN's (layer 0: stream 0),
//!                  the FFN by this attention's; the head by the last FFN's
//!   engram:        before layers 1 and 14: the 4-gram hash of the tokens (3 n-gram sizes x 8 heads = 24 table rows a
//!                  token, read from the official shards on disk), wkv [6144 -> 5 x 5120] to a key a stream and a value,
//!                  each stream gated by its normalized dot with its key
//!   attention:     q through a 1280 LoRA (q_a, its norm, q_b) to 64 heads of 512 (no per-head norm); one key head kv
//!                  [512] (normed, the last 64 rotated, fp8-simulated) that is the value too, in a 128-position window;
//!                  by the layer's compress ratio (0: layers 0, 1; 2: 2-19; 1: 20-39) also compressed keys:
//!                    only the kv sources (2, 8, 14, 20) compress - ratio 2 pools pairs (a softmax gate a value), ratio 1
//!                    projects each token - normed, rotated at the group's first position, fp4-simulated; the layers
//!                    after a source read its cache
//!                    the index sources (2, 8, 14, 20, 24, 28, 32, 36) score them (32 heads of 128; keys from the
//!                    source's latent through wk), the layers after a source reuse its top 512; layer 20 also keeps
//!                    each query's best 2048 blocks of 8, inside which 24-36 pick
//!                  a softmax with a learned sink a head; the output rotated back, fp8-simulated, 8 groups through o_a
//!                  [1024 of 4096] and all through o_b
//!   feed-forward:  MoE every layer: 384 experts of 2304 (VQ: 8 weights an index into the layer's E4M3 codebook, an f16
//!                  gain a row; layers 0-13 13-bit, 14-39 12-bit), top-6 by sqrt(softplus) + a bias, normalized x 1.5,
//!                  one shared expert (q4_K); SwiGLU clamped at 10
//!   head:          the streams collapsed by the last FFN's mix, the norm, the output matrix
//!   draft (MTP):   three DSpark towers (mtp.0-2), blocks of their own over [token, 4 noise tokens]: window-only
//!                  attention whose window holds main_x = main_norm(main_proj(the stream means at layers 37-39's input));
//!                  128 experts top-3 (VQ 12-bit); the head's logits plus a Markov bias from the previous token
//! ```

use nextsycl_gguf::{GType, Gguf, Tensor, Value};

use crate::{Error, ModelResult as Result};

/// The geometry, from metadata and the tensors' shapes.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub n_embd: u64,
    pub n_vocab: u64,
    pub n_layer: u64,
    pub n_expert: u64,
    pub n_expert_used: u64,
    pub ffn_expert: u64,
    pub expert_scale: f64,
    pub swiglu_limit: f64,
    pub rms_eps: f64,
    pub n_head: u64,
    pub head_dim: u64,
    pub rope_dim: u64,
    pub q_lora: u64,
    pub o_groups: u64,
    pub o_lora: u64,
    pub window: u64,
    pub idx_heads: u64,
    pub idx_dim: u64,
    pub idx_top_k: u64,
    pub hc: u64,
    pub hc_iters: u64,
    pub hc_eps: f64,
    pub rope_base: f64,
    pub comp_rope_base: f64,
    pub yarn_factor: f64,
    pub yarn_orig_ctx: u64,
    pub yarn_beta_fast: f64,
    pub yarn_beta_slow: f64,
    pub ctx: u64,
    /// per block (the towers' too): 0, 1 or 2
    pub ratio: Vec<u64>,
    pub kv_sources: Vec<u64>,
    pub index_sources: Vec<u64>,
    pub cand_source: Option<u64>,
    pub cand_top: u64,
    pub cand_block: u64,
    pub engram_layers: Vec<u64>,
    pub engram_heads: u64,
    pub engram_dim: u64,
    pub engram_ngram: u64,
    /// the draft towers (blocks n_layer..n_layer + n_mtp) and theirs
    pub n_mtp: u64,
    pub mtp_expert: u64,
    pub mtp_expert_used: u64,
    pub mtp_block: u64,
    pub mtp_noise: u32,
    pub mtp_targets: Vec<u64>,
    pub markov_rank: u64,
}

impl Geometry {
    pub fn ratio(&self, l: u64) -> u64 {
        self.ratio.get(l as usize).copied().unwrap_or(0)
    }
    /// the kv source whose compressed keys layer l reads (the last source at or before it)
    pub fn kv_source(&self, l: u64) -> Option<u64> {
        if self.ratio(l) == 0 || l >= self.n_layer {
            return None;
        }
        self.kv_sources.iter().rev().copied().find(|s| *s <= l)
    }
    /// the index source whose picks layer l uses
    pub fn index_source(&self, l: u64) -> Option<u64> {
        if self.ratio(l) == 0 || l >= self.n_layer {
            return None;
        }
        self.index_sources.iter().rev().copied().find(|s| *s <= l)
    }
    pub fn is_mtp(&self, l: u64) -> bool {
        l >= self.n_layer && l < self.n_layer + self.n_mtp
    }
    pub fn is_engram(&self, l: u64) -> bool {
        self.engram_layers.contains(&l)
    }
    /// experts of block l (a tower's 128, the main layers' 384) and its top-k
    pub fn experts(&self, l: u64) -> (u64, u64) {
        if self.is_mtp(l) { (self.mtp_expert, self.mtp_expert_used) } else { (self.n_expert, self.n_expert_used) }
    }
    /// rows an Engram lookup reads a token and layer
    pub fn engram_cols(&self) -> u64 {
        (self.engram_ngram - 1) * self.engram_heads
    }
}

/// The rotation a layer's (and its compressor's and indexer's) rope uses: (base, freq_scale, ext_factor, mscale, corr0,
/// corr1)
#[derive(Clone, Copy, Debug)]
pub struct Rope {
    pub base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub mscale: f32,
    pub corr0: f32,
    pub corr1: f32,
}

/// A tensor's role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    TokenEmbd,
    OutputNorm,
    Output,
    EngramTokenMap,
    EngramMult,
    EngramPrimes,
    EngramOffsets,
    MtpMainProj,
    MtpMainNorm,
    MtpMarkovEmbd,
    MtpMarkovHead,
    MtpConfidence,
    MtpOutNorm,
    // every block
    AttnNorm,
    FfnNorm,
    HcAttnFn,
    HcAttnBase,
    HcAttnScale,
    HcFfnFn,
    HcFfnBase,
    HcFfnScale,
    QA,
    QANorm,
    QB,
    Kv,
    KvNorm,
    Sinks,
    OutA,
    OutB,
    // the kv sources' compressor and the indexer's keys
    CompKv,
    CompGate,
    CompNorm,
    IdxWk,
    IdxKNorm,
    // the index sources'
    IdxQB,
    IdxProj,
    // MoE
    Router,
    RouterBias,
    Experts,
    ShGate,
    ShUp,
    ShDown,
    // the Engram layers'
    EngramWkv,
    EngramQ,
    EngramK,
}

/// What a role belongs to, for the memory plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// embedding, the head, the Engram's tables of constants
    Edges,
    /// attention, compressors, indexer, norms, hyper-connections, the Engram's projections
    Attention,
    /// the router and the shared expert
    Shared,
    /// the routed experts
    Experts,
    /// the draft towers and their heads
    Draft,
}

impl Role {
    pub fn group(self) -> Group {
        use Role::*;
        match self {
            TokenEmbd | OutputNorm | Output | EngramTokenMap | EngramMult | EngramPrimes | EngramOffsets => Group::Edges,
            MtpMainProj | MtpMainNorm | MtpMarkovEmbd | MtpMarkovHead | MtpConfidence | MtpOutNorm => Group::Draft,
            Router | RouterBias | ShGate | ShUp | ShDown => Group::Shared,
            Experts => Group::Experts,
            _ => Group::Attention,
        }
    }

    /// The tensor's name suffix (after `blk.N.` / `mtp.T.`; the whole name for a global role).
    pub fn name(self) -> &'static str {
        use Role::*;
        match self {
            TokenEmbd => "token_embd.weight",
            OutputNorm => "output_norm.weight",
            Output => "output.weight",
            EngramTokenMap => "engram.token_map",
            EngramMult => "engram.multipliers",
            EngramPrimes => "engram.primes",
            EngramOffsets => "engram.offsets",
            MtpMainProj => "mtp.main_proj.weight",
            MtpMainNorm => "mtp.main_norm.weight",
            MtpMarkovEmbd => "mtp.markov_embd.weight",
            MtpMarkovHead => "mtp.markov_head.weight",
            MtpConfidence => "mtp.confidence.weight",
            MtpOutNorm => "mtp.out_norm.weight",
            AttnNorm => "attn_norm.weight",
            FfnNorm => "ffn_norm.weight",
            HcAttnFn => "hc_attn_fn.weight",
            HcAttnBase => "hc_attn_base.weight",
            HcAttnScale => "hc_attn_scale.weight",
            HcFfnFn => "hc_ffn_fn.weight",
            HcFfnBase => "hc_ffn_base.weight",
            HcFfnScale => "hc_ffn_scale.weight",
            QA => "attn_q_a.weight",
            QANorm => "attn_q_a_norm.weight",
            QB => "attn_q_b.weight",
            Kv => "attn_kv.weight",
            KvNorm => "attn_kv_a_norm.weight",
            Sinks => "attn_sinks.weight",
            OutA => "attn_output_a.weight",
            OutB => "attn_output_b.weight",
            CompKv => "attn_compressor_kv.weight",
            CompGate => "attn_compressor_gate.weight",
            CompNorm => "attn_compressor_norm.weight",
            IdxWk => "indexer.wk.weight",
            IdxKNorm => "indexer.k_norm.weight",
            IdxQB => "indexer.attn_q_b.weight",
            IdxProj => "indexer.proj.weight",
            Router => "ffn_gate_inp.weight",
            RouterBias => "exp_probs_b.bias",
            Experts => "ffn_exps_vq.blob",
            ShGate => "ffn_gate_shexp.weight",
            ShUp => "ffn_up_shexp.weight",
            ShDown => "ffn_down_shexp.weight",
            EngramWkv => "engram_wkv.weight",
            EngramQ => "engram_q.weight",
            EngramK => "engram_k.weight",
        }
    }

    pub fn global(self) -> bool {
        matches!(self.group(), Group::Edges | Group::Draft)
    }
}

const GLOBALS: [Role; 13] = [Role::TokenEmbd, Role::OutputNorm, Role::Output, Role::EngramTokenMap, Role::EngramMult, Role::EngramPrimes, Role::EngramOffsets,
                             Role::MtpMainProj, Role::MtpMainNorm, Role::MtpMarkovEmbd, Role::MtpMarkovHead, Role::MtpConfidence, Role::MtpOutNorm];
const COMMON: [Role; 16] = [Role::AttnNorm, Role::FfnNorm, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale,
                            Role::QA, Role::QANorm, Role::QB, Role::Kv, Role::KvNorm, Role::Sinks, Role::OutA, Role::OutB];
const MOE: [Role; 6] = [Role::Router, Role::RouterBias, Role::Experts, Role::ShGate, Role::ShUp, Role::ShDown];

/// A model file, resolved.
pub struct Model<'g> {
    pub file: &'g Gguf,
    pub g: Geometry,
}

fn meta_u(f: &Gguf, k: &str) -> Result<u64> {
    f.arch_meta(k).and_then(Value::as_u64).ok_or_else(|| Error(format!("metadata {k} is missing")))
}
fn meta_f(f: &Gguf, k: &str, default: Option<f64>) -> Result<f64> {
    f.arch_meta(k).and_then(|v| v.as_f64().or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_f64))).or(default)
        .ok_or_else(|| Error(format!("metadata {k} is missing")))
}
fn meta_list(f: &Gguf, k: &str) -> Result<Vec<u64>> {
    f.arch_meta(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).ok_or_else(|| Error(format!("metadata {k} is missing")))
}

impl<'g> Model<'g> {
    pub fn open(file: &'g Gguf) -> Result<Model<'g>> {
        if file.arch_meta("variant").and_then(Value::as_str) != Some("v41_flash") {
            return Err(Error("not a DeepSeek-V4.1-Flash file (deepseek4.variant v41_flash)".into()));
        }
        let gating = file.arch_meta("expert_gating_func");
        if !matches!(gating, Some(Value::Str(s)) if s == "sqrtsoftplus") && gating.and_then(Value::as_u64) != Some(4) {
            return Err(Error(format!("expert_gating_func {gating:?}: the engine takes sqrt(softplus)")));
        }
        let n_layer = meta_u(file, "block_count")?;
        let n_mtp = meta_u(file, "mtp.tower_count").unwrap_or(0);
        let ratio = meta_list(file, "attention.compress_ratios")?;
        if let Some(r) = ratio.iter().find(|r| **r > 2) {
            return Err(Error(format!("a compress ratio of {r}: the engine takes 0, 1 and 2")));
        }
        let cand = meta_u(file, "attention.candidate.source_layer").ok();
        let g = Geometry {
            n_embd: meta_u(file, "embedding_length")?,
            n_vocab: meta_u(file, "vocab_size")?,
            n_layer,
            n_expert: meta_u(file, "expert_count")?,
            n_expert_used: meta_u(file, "expert_used_count")?,
            ffn_expert: meta_u(file, "expert_feed_forward_length")?,
            expert_scale: meta_f(file, "expert_weights_scale", None)?,
            swiglu_limit: meta_f(file, "swiglu_clamp_exp", Some(10.0))?,
            rms_eps: meta_f(file, "attention.layer_norm_rms_epsilon", None)?,
            n_head: meta_u(file, "attention.head_count")?,
            head_dim: meta_u(file, "attention.key_length")?,
            rope_dim: meta_u(file, "rope.dimension_count")?,
            q_lora: meta_u(file, "attention.q_lora_rank")?,
            o_groups: meta_u(file, "attention.output_group_count")?,
            o_lora: meta_u(file, "attention.output_lora_rank")?,
            window: meta_u(file, "attention.sliding_window")?,
            idx_heads: meta_u(file, "attention.indexer.head_count")?,
            idx_dim: meta_u(file, "attention.indexer.key_length")?,
            idx_top_k: meta_u(file, "attention.indexer.top_k")?,
            hc: meta_u(file, "hyper_connection.count")?,
            hc_iters: meta_u(file, "hyper_connection.sinkhorn_iterations")?,
            hc_eps: meta_f(file, "hyper_connection.epsilon", None)?,
            rope_base: meta_f(file, "rope.freq_base", None)?,
            comp_rope_base: meta_f(file, "attention.compress_rope_freq_base", None)?,
            yarn_factor: meta_f(file, "rope.scaling.factor", Some(1.0))?,
            yarn_orig_ctx: meta_u(file, "rope.scaling.original_context_length").unwrap_or(0),
            yarn_beta_fast: meta_f(file, "rope.scaling.yarn_beta_fast", Some(32.0))?,
            yarn_beta_slow: meta_f(file, "rope.scaling.yarn_beta_slow", Some(1.0))?,
            ctx: meta_u(file, "context_length")?,
            ratio,
            kv_sources: meta_list(file, "attention.kv_source_layers")?,
            index_sources: meta_list(file, "attention.index_source_layers")?,
            cand_source: cand,
            cand_top: meta_u(file, "attention.candidate.topk_blocks").unwrap_or(0),
            cand_block: meta_u(file, "attention.candidate.block_size").unwrap_or(1),
            engram_layers: meta_list(file, "engram.layer_ids").unwrap_or_default(),
            engram_heads: meta_u(file, "engram.head_count").unwrap_or(0),
            engram_dim: meta_u(file, "engram.head_dim").unwrap_or(0),
            engram_ngram: meta_u(file, "engram.max_ngram_size").unwrap_or(1),
            n_mtp,
            mtp_expert: meta_u(file, "mtp.expert_count").unwrap_or(0),
            mtp_expert_used: meta_u(file, "mtp.expert_used_count").unwrap_or(0),
            mtp_block: meta_u(file, "mtp.block_size").unwrap_or(0),
            mtp_noise: meta_u(file, "mtp.noise_token_id").unwrap_or(0) as u32,
            mtp_targets: meta_list(file, "mtp.target_layers").unwrap_or_default(),
            markov_rank: meta_u(file, "mtp.markov_rank").unwrap_or(0),
        };
        if g.hc != 4 || g.head_dim != 512 {
            return Err(Error(format!("{} mHC streams, heads of {}: the engine takes 4 and 512", g.hc, g.head_dim)));
        }
        for &s in g.index_sources.iter() {
            let src = g.kv_sources.iter().rev().copied().find(|k| *k <= s);
            if g.ratio(s) == 0 || src.is_none_or(|k| g.ratio(k) != g.ratio(s)) {
                return Err(Error(format!("index source {s}: no kv source of its ratio before it")));
            }
        }
        for l in 0..n_layer {
            if g.ratio(l) > 0 && (g.kv_source(l).is_none_or(|k| g.ratio(k) != g.ratio(l)) || g.index_source(l).is_none()) {
                return Err(Error(format!("layer {l}: compressed (ratio {}) without a kv / index source of its ratio before it", g.ratio(l))));
            }
        }
        Ok(Model { file, g })
    }

    /// A block's tensor name prefix: `blk.N.` or a tower's `mtp.T.`
    pub fn prefix(&self, l: u64) -> String {
        if l >= self.g.n_layer { format!("mtp.{}.", l - self.g.n_layer) } else { format!("blk.{l}.") }
    }

    /// A tensor by role (`layer` ignored for global roles).
    pub fn tensor(&self, l: u64, r: Role) -> Option<&'g Tensor> {
        if r.global() { self.file.tensor(r.name()) } else { self.file.tensor(&format!("{}{}", self.prefix(l), r.name())) }
    }

    /// The roles block `l` has.
    pub fn roles(&self, l: u64) -> Vec<Role> {
        let g = &self.g;
        let mut v = COMMON.to_vec();
        if g.kv_sources.contains(&l) {
            v.extend([Role::CompKv, Role::CompNorm, Role::IdxWk, Role::IdxKNorm]);
            if g.ratio(l) > 1 {
                v.push(Role::CompGate);
            }
        }
        if g.index_sources.contains(&l) {
            v.extend([Role::IdxQB, Role::IdxProj]);
        }
        v.extend(MOE);
        if g.is_engram(l) {
            v.extend([Role::EngramWkv, Role::EngramQ, Role::EngramK]);
        }
        v
    }

    /// The shape a role must have (row-major, outermost first) in block l.
    pub fn shape(&self, l: u64, r: Role) -> Vec<u64> {
        use Role::*;
        let g = &self.g;
        let (d, hc, hd) = (g.n_embd, g.hc, g.head_dim);
        let (ne, _) = g.experts(l);
        let ecols = g.engram_cols();
        match r {
            TokenEmbd | Output => vec![g.n_vocab, d],
            OutputNorm | AttnNorm | FfnNorm | MtpMainNorm | MtpOutNorm => vec![d],
            EngramTokenMap => vec![g.n_vocab],
            EngramMult => vec![g.engram_layers.len() as u64, g.engram_ngram],
            EngramPrimes => vec![g.engram_layers.len() as u64, g.engram_ngram - 1, g.engram_heads],
            EngramOffsets => vec![g.engram_layers.len() as u64, ecols],
            MtpMainProj => vec![d, d * g.mtp_targets.len() as u64],
            MtpMarkovEmbd | MtpMarkovHead => vec![g.n_vocab, g.markov_rank],
            MtpConfidence => vec![1, d + g.markov_rank],
            HcAttnFn | HcFfnFn => vec![hc * 2 + hc * hc, hc * d],
            HcAttnBase | HcFfnBase => vec![hc * 2 + hc * hc],
            HcAttnScale | HcFfnScale => vec![3],
            QA => vec![g.q_lora, d],
            QANorm => vec![g.q_lora],
            QB => vec![g.n_head * hd, g.q_lora],
            Kv | CompKv | CompGate => vec![hd, d],
            KvNorm | CompNorm => vec![hd],
            Sinks => vec![g.n_head],
            OutA => vec![g.o_groups * g.o_lora, g.n_head * hd / g.o_groups],
            OutB => vec![d, g.o_groups * g.o_lora],
            IdxWk => vec![g.idx_dim, hd],
            IdxKNorm => vec![g.idx_dim],
            IdxQB => vec![g.idx_heads * g.idx_dim, g.q_lora],
            IdxProj => vec![g.idx_heads, d],
            Router => vec![ne, d],
            RouterBias => vec![ne],
            Experts => self.tensor(l, r).map_or(vec![0], |t| t.shape.clone()),
            ShGate | ShUp => vec![g.ffn_expert, d],
            ShDown => vec![d, g.ffn_expert],
            EngramWkv => vec![(hc + 1) * d, ecols * g.engram_dim],
            EngramQ | EngramK => vec![hc, d],
        }
    }

    /// Every role present with the shape the runtime needs, and no tensor left over. All problems.
    pub fn check(&self) -> Vec<String> {
        let mut errs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for r in GLOBALS {
            if r.group() == Group::Draft && self.g.n_mtp == 0 {
                continue;
            }
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
            None => errs.push(format!("block {l}: {:?} ({}) is missing", r, r.name())),
            Some(t) => {
                seen.insert(t.name.as_str());
                let want = self.shape(l, r);
                if t.shape != want {
                    errs.push(format!("{}: shape {:?}, the runtime needs {:?}", t.name, t.shape, want));
                }
                if r == Role::Experts && t.ty != GType::VqBlob {
                    errs.push(format!("{}: type {}, the runtime needs a VQ blob", t.name, t.ty.name()));
                }
            }
        }
    }

    /// Bytes by group and type, for `info`.
    pub fn bytes(&self) -> std::collections::BTreeMap<(Group, GType), (u64, u64)> {
        let mut m = std::collections::BTreeMap::new();
        let mut add = |r: Role, t: &Tensor| {
            let e = m.entry((r.group(), t.ty)).or_insert((0u64, 0u64));
            e.0 += t.bytes;
            e.1 += 1;
        };
        for r in GLOBALS {
            if let Some(t) = self.tensor(0, r) {
                add(r, t);
            }
        }
        for l in 0..self.g.n_layer + self.g.n_mtp {
            for r in self.roles(l) {
                if let Some(t) = self.tensor(l, r) {
                    add(if l >= self.g.n_layer { Role::MtpMainProj } else { r }, t);
                }
            }
        }
        m
    }

    /// The rotation of block l's queries and keys: a compressed layer (and its compressor and indexer) at the
    /// compress base with YaRN, the others (and the towers) plainly at the base
    pub fn rope(&self, l: u64) -> Rope {
        let g = &self.g;
        if g.ratio(l) == 0 {
            return Rope { base: g.rope_base as f32, freq_scale: 1.0, ext_factor: 0.0, mscale: 1.0, corr0: 0.0, corr1: 0.0 };
        }
        let base = g.comp_rope_base as f32;
        let yarn = g.yarn_factor > 1.0;
        let freq_scale = if yarn { 1.0 / g.yarn_factor as f32 } else { 1.0 };
        let ext_factor = if yarn { 1.0 } else { 0.0 };
        let n = g.rope_dim as f32;
        let corr = |beta: f64| n * (g.yarn_orig_ctx as f32 / (beta as f32 * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
        let c0 = corr(g.yarn_beta_fast).floor().max(0.0);
        let c1 = corr(g.yarn_beta_slow).ceil().min(n - 1.0);
        Rope { base, freq_scale, ext_factor, mscale: 1.0, corr0: c0, corr1: c1 }
    }
}
