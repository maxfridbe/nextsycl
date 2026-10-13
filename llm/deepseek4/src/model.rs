//! DeepSeek-V4-Flash (`deepseek4`): 43 layers, hidden 4096, the residual as 4 mHC streams.
//!
//! ```text
//!   every layer:   hyper-connections around both halves (as GLM-5.3's: hc_*_fn [24, 4 x hidden], Sinkhorn), then
//!   attention:     q through a 1024 LoRA (q_a, its norm, q_b) to 64 heads of 512, each RMS-normed; one key head
//!                  kv [512] (normed) that is the value too; the last 64 values of every head rotated (YaRN at the
//!                  compressed layers' base 160000, plain RoPE at 10000 on layers 0, 1); the keys a query reads:
//!                    the raw ones of the last 128 positions, and by the layer's compress ratio
//!                    0   (layers 0, 1): none more
//!                    4   (CSA: 2, 4, .., 42): a block per 4 tokens - pooled over 8 (the previous block's tokens too,
//!                        on the first half of their projections), each value a softmax of gate + ape over the tokens
//!                        weighting kv - normed and rotated at the block's start; a lightning indexer (64 heads of 128
//!                        from q's LoRA, keys from its own such compressor) keeps each query's best 512 of them
//!                    128 (HCA: 3, 5, .., 41): a block per 128 tokens the same way (no overlap), every one read
//!                  a softmax with a learned sink per head; the output rotated back, then grouped: 8 groups of 8
//!                  heads each through o_a [1024 of 4096], all of them through o_b [4096 of 8192]
//!   feed-forward:  MoE every layer: 256 experts of 2048, top-6 by sqrt(softplus(router)) + a bias (layers 0-2: the
//!                  experts a table gives the token), weights normalized x 1.5, one shared expert; SwiGLU clamped at 10
//!   head:          the streams combined by their own mHC weights (output_hc_*), the norm, the output matrix
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
    pub n_expert_shared: u64,
    pub ffn_expert: u64,
    pub expert_scale: f64,
    pub expert_norm: bool,
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
    pub hash_layers: u64,
    pub rope_base: f64,
    pub comp_rope_base: f64,
    pub yarn_factor: f64,
    pub yarn_orig_ctx: u64,
    pub yarn_beta_fast: f64,
    pub yarn_beta_slow: f64,
    /// per layer: 0, 4 or 128
    pub ratio: Vec<u64>,
}

impl Geometry {
    pub fn ratio(&self, layer: u64) -> u64 {
        self.ratio.get(layer as usize).copied().unwrap_or(0)
    }
    pub fn is_hash(&self, layer: u64) -> bool {
        layer < self.hash_layers
    }
}

/// The rotation a layer's (or its compressor's) rope uses: (base, freq_scale, ext_factor, mscale, corr0, corr1)
#[derive(Clone, Copy, Debug)]
pub struct Rope {
    pub base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub mscale: f32,
    pub corr0: f32,
    pub corr1: f32,
}

/// A tensor's role, the rest global.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    TokenEmbd,
    OutputNorm,
    Output,
    HeadHcFn,
    HeadHcBase,
    HeadHcScale,
    // every layer
    AttnNorm,
    FfnNorm,
    HcAttnFn,
    HcAttnBase,
    HcAttnScale,
    HcFfnFn,
    HcFfnBase,
    HcFfnScale,
    // attention
    QA,
    QANorm,
    QB,
    Kv,
    KvNorm,
    Sinks,
    OutA,
    OutB,
    // the compressor (ratio 4 / 128)
    CompKv,
    CompGate,
    CompApe,
    CompNorm,
    // the indexer (ratio 4)
    IdxQB,
    IdxProj,
    IdxCompKv,
    IdxCompGate,
    IdxCompApe,
    IdxCompNorm,
    // MoE
    Router,
    RouterBias,
    HashTable,
    ExpGate,
    ExpUp,
    ExpDown,
    ShGate,
    ShUp,
    ShDown,
}

/// What a role belongs to, for the memory plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// embedding and the head
    Edges,
    /// attention, compressors, indexer, norms, hyper-connections
    Attention,
    /// the router and the shared expert: on every token, always resident
    Shared,
    /// the routed experts: 6 of 256 per token, what does not fit VRAM swaps
    Experts,
}

impl Role {
    pub fn group(self) -> Group {
        use Role::*;
        match self {
            TokenEmbd | OutputNorm | Output | HeadHcFn | HeadHcBase | HeadHcScale => Group::Edges,
            Router | RouterBias | HashTable | ShGate | ShUp | ShDown => Group::Shared,
            ExpGate | ExpUp | ExpDown => Group::Experts,
            _ => Group::Attention,
        }
    }

    /// The tensor's name suffix in a file (after `blk.N.`; the whole name for a global role).
    pub fn name(self) -> &'static str {
        use Role::*;
        match self {
            TokenEmbd => "token_embd.weight",
            OutputNorm => "output_norm.weight",
            Output => "output.weight",
            HeadHcFn => "output_hc_fn.weight",
            HeadHcBase => "output_hc_base.weight",
            HeadHcScale => "output_hc_scale.weight",
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
            CompApe => "attn_compressor_ape.weight",
            CompNorm => "attn_compressor_norm.weight",
            IdxQB => "indexer.attn_q_b.weight",
            IdxProj => "indexer.proj.weight",
            IdxCompKv => "indexer_compressor_kv.weight",
            IdxCompGate => "indexer_compressor_gate.weight",
            IdxCompApe => "indexer_compressor_ape.weight",
            IdxCompNorm => "indexer_compressor_norm.weight",
            Router => "ffn_gate_inp.weight",
            RouterBias => "exp_probs_b.bias",
            HashTable => "ffn_gate_tid2eid.weight",
            ExpGate => "ffn_gate_exps.weight",
            ExpUp => "ffn_up_exps.weight",
            ExpDown => "ffn_down_exps.weight",
            ShGate => "ffn_gate_shexp.weight",
            ShUp => "ffn_up_shexp.weight",
            ShDown => "ffn_down_shexp.weight",
        }
    }

    fn global(self) -> bool {
        matches!(self, Role::TokenEmbd | Role::OutputNorm | Role::Output | Role::HeadHcFn | Role::HeadHcBase | Role::HeadHcScale)
    }
}

const COMMON: [Role; 16] = [Role::AttnNorm, Role::FfnNorm, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale,
                            Role::QA, Role::QANorm, Role::QB, Role::Kv, Role::KvNorm, Role::Sinks, Role::OutA, Role::OutB];
const COMP: [Role; 4] = [Role::CompKv, Role::CompGate, Role::CompApe, Role::CompNorm];
const IDX: [Role; 6] = [Role::IdxQB, Role::IdxProj, Role::IdxCompKv, Role::IdxCompGate, Role::IdxCompApe, Role::IdxCompNorm];
const MOE: [Role; 7] = [Role::Router, Role::ExpGate, Role::ExpUp, Role::ExpDown, Role::ShGate, Role::ShUp, Role::ShDown];

/// A model file, resolved: its geometry and every tensor by (layer, role).
pub struct Model<'g> {
    pub file: &'g Gguf,
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
        let n_layer = meta_u(file, &["block_count"])?;
        let ratios = file.arch_meta("attention.compress_ratios").and_then(Value::as_array).ok_or("attention.compress_ratios is missing")?;
        let ratio: Vec<u64> = (0..n_layer as usize).map(|i| ratios.get(i).and_then(Value::as_u64).unwrap_or(0)).collect();
        if let Some(r) = ratio.iter().find(|r| !matches!(r, 0 | 4 | 128)) {
            return Err(Error(format!("a compress ratio of {r}: the engine takes 0, 4 and 128")));
        }
        let shape = |name: &str| file.tensor(name).map(|t| t.shape.clone()).ok_or_else(|| Error(format!("{name} is missing")));
        let gating = meta_u(file, &["expert_gating_func"])?;
        if gating != 4 {
            return Err(Error(format!("expert_gating_func {gating}: the engine takes sqrt(softplus) (4)")));
        }
        let g = Geometry {
            n_embd: meta_u(file, &["embedding_length"])?,
            n_vocab: meta_u(file, &["vocab_size"]).or_else(|_| shape(Role::TokenEmbd.name()).map(|s| s[0]))?,
            n_layer,
            n_expert: meta_u(file, &["expert_count"])?,
            n_expert_used: meta_u(file, &["expert_used_count"])?,
            n_expert_shared: meta_u(file, &["expert_shared_count"])?,
            ffn_expert: meta_u(file, &["expert_feed_forward_length"])?,
            expert_scale: meta_f(file, &["expert_weights_scale"], None)?,
            expert_norm: file.arch_meta("expert_weights_norm").and_then(Value::as_u64).is_none_or(|x| x != 0),
            swiglu_limit: meta_f(file, &["swiglu_clamp_exp"], Some(10.0))?,
            rms_eps: meta_f(file, &["attention.layer_norm_rms_epsilon"], None)?,
            n_head: meta_u(file, &["attention.head_count"])?,
            head_dim: meta_u(file, &["attention.key_length"])?,
            rope_dim: meta_u(file, &["rope.dimension_count"])?,
            q_lora: meta_u(file, &["attention.q_lora_rank"])?,
            o_groups: meta_u(file, &["attention.output_group_count"])?,
            o_lora: meta_u(file, &["attention.output_lora_rank"])?,
            window: meta_u(file, &["attention.sliding_window"])?,
            idx_heads: meta_u(file, &["attention.indexer.head_count"])?,
            idx_dim: meta_u(file, &["attention.indexer.key_length"])?,
            idx_top_k: meta_u(file, &["attention.indexer.top_k"])?,
            hc: meta_u(file, &["hyper_connection.count"])?,
            hc_iters: meta_u(file, &["hyper_connection.sinkhorn_iterations"])?,
            hc_eps: meta_f(file, &["hyper_connection.epsilon"], None)?,
            hash_layers: meta_u(file, &["hash_layer_count"]).unwrap_or(0),
            rope_base: meta_f(file, &["rope.freq_base"], None)?,
            comp_rope_base: meta_f(file, &["attention.compress_rope_freq_base"], None)?,
            yarn_factor: meta_f(file, &["rope.scaling.factor"], Some(1.0))?,
            yarn_orig_ctx: meta_u(file, &["rope.scaling.original_context_length"]).unwrap_or(0),
            yarn_beta_fast: meta_f(file, &["rope.scaling.yarn_beta_fast"], Some(32.0))?,
            yarn_beta_slow: meta_f(file, &["rope.scaling.yarn_beta_slow"], Some(1.0))?,
            ratio,
        };
        if g.hc != 4 {
            return Err(Error(format!("{} mHC streams: the engine takes 4", g.hc)));
        }
        if g.head_dim != 512 {
            return Err(Error(format!("heads of {}: the engine's attention takes 512", g.head_dim)));
        }
        Ok(Model { file, g })
    }

    /// A tensor by role (`layer` ignored for global roles).
    pub fn tensor(&self, layer: u64, r: Role) -> Option<&'g Tensor> {
        if r.global() {
            self.file.tensor(r.name())
        } else {
            self.file.tensor(&format!("blk.{layer}.{}", r.name()))
        }
    }

    /// The roles block `layer` has.
    pub fn roles(&self, layer: u64) -> Vec<Role> {
        let g = &self.g;
        let mut v = COMMON.to_vec();
        if g.ratio(layer) != 0 {
            v.extend(COMP);
        }
        if g.ratio(layer) == 4 {
            v.extend(IDX);
        }
        v.extend(MOE);
        v.push(if g.is_hash(layer) { Role::HashTable } else { Role::RouterBias });
        v
    }

    /// The shape a role must have (row-major, outermost first) in layer `l`.
    pub fn shape(&self, l: u64, r: Role) -> Vec<u64> {
        use Role::*;
        let g = &self.g;
        let (d, hc) = (g.n_embd, g.hc);
        let coff = if g.ratio(l) == 4 { 2 } else { 1 };
        match r {
            TokenEmbd | Output => vec![g.n_vocab, d],
            OutputNorm | AttnNorm | FfnNorm => vec![d],
            HeadHcFn => vec![hc, hc * d],
            HeadHcBase => vec![hc],
            HeadHcScale => vec![1],
            HcAttnFn | HcFfnFn => vec![hc * 2 + hc * hc, hc * d],
            HcAttnBase | HcFfnBase => vec![hc * 2 + hc * hc],
            HcAttnScale | HcFfnScale => vec![3],
            QA => vec![g.q_lora, d],
            QANorm => vec![g.q_lora],
            QB => vec![g.n_head * g.head_dim, g.q_lora],
            Kv => vec![g.head_dim, d],
            KvNorm => vec![g.head_dim],
            Sinks => vec![g.n_head],
            OutA => vec![g.o_groups * g.o_lora, g.n_head * g.head_dim / g.o_groups],
            OutB => vec![d, g.o_groups * g.o_lora],
            CompKv | CompGate => vec![coff * g.head_dim, d],
            CompApe => vec![g.ratio(l), coff * g.head_dim],
            CompNorm => vec![g.head_dim],
            IdxQB => vec![g.idx_heads * g.idx_dim, g.q_lora],
            IdxProj => vec![g.idx_heads, d],
            IdxCompKv | IdxCompGate => vec![2 * g.idx_dim, d],
            IdxCompApe => vec![4, 2 * g.idx_dim],
            IdxCompNorm => vec![g.idx_dim],
            Router => vec![g.n_expert, d],
            RouterBias => vec![g.n_expert],
            HashTable => vec![g.n_vocab, g.n_expert_used],
            ExpGate | ExpUp => vec![g.n_expert, g.ffn_expert, d],
            ExpDown => vec![g.n_expert, d, g.ffn_expert],
            ShGate | ShUp => vec![g.ffn_expert * g.n_expert_shared, d],
            ShDown => vec![d, g.ffn_expert * g.n_expert_shared],
        }
    }

    /// Every block's every role: present and of the right shape; no tensor in the file left over. All problems,
    /// not the first.
    pub fn check(&self) -> Vec<String> {
        let mut errs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for r in [Role::TokenEmbd, Role::OutputNorm, Role::Output, Role::HeadHcFn, Role::HeadHcBase, Role::HeadHcScale] {
            self.check_one(0, r, &mut errs, &mut seen);
        }
        for l in 0..self.g.n_layer {
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
        for r in [Role::TokenEmbd, Role::OutputNorm, Role::Output, Role::HeadHcFn, Role::HeadHcBase, Role::HeadHcScale] {
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

    /// The rotation of layer `l`'s queries and raw keys (`comp`: its compressor's and indexer's - the same at a
    /// compressed layer): a compressed layer rotates at the compress base with YaRN, the others plainly at the base.
    /// ggml's mscale (attn_factor x (1 + 0.1 ln(1 / freq_scale)) with YaRN) is 1: llama.cpp passes attn_factor
    /// 1 / (1 + 0.1 ln(1 / freq_scale))
    pub fn rope(&self, l: u64) -> Rope {
        let g = &self.g;
        if g.ratio(l) == 0 {
            return Rope { base: g.rope_base as f32, freq_scale: 1.0, ext_factor: 0.0, mscale: 1.0, corr0: 0.0, corr1: 0.0 };
        }
        let base = g.comp_rope_base as f32;
        let yarn = g.yarn_factor > 1.0;
        let freq_scale = if yarn { 1.0 / g.yarn_factor as f32 } else { 1.0 };
        let ext_factor = if yarn { 1.0 } else { 0.0 };
        // ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, base, beta_fast, beta_slow)
        let n = g.rope_dim as f32;
        let corr = |beta: f64| n * (g.yarn_orig_ctx as f32 / (beta as f32 * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
        let c0 = corr(g.yarn_beta_fast).floor().max(0.0);
        let c1 = corr(g.yarn_beta_slow).ceil().min(n - 1.0);
        Rope { base, freq_scale, ext_factor, mscale: 1.0, corr0: c0, corr1: c1 }
    }
}
