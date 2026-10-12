//! DeepSeek-V3 / R1 (`deepseek2`): 61 layers, hidden 7168, 129,280 tokens.
//!
//! ```text
//!   every layer:   x += MLA(rms(x)); x += FFN(rms(x))
//!   MLA:           q through a 1536 LoRA (q_a -> norm -> q_b: 128 heads of 128 + 64), keys and values through a
//!                  shared 512 latent (kv_a -> [512 | 64]: the latent normed, the 64 a key part every head shares);
//!                  the 64s rotated by YaRN's RoPE (factor 40 from 4,096); per head k = [k_b . latent | rope],
//!                  v = v_b . latent - absorbed: the cache holds [latent | rope] (576), the queries [k_b^T q | rope]
//!   feed-forward:  dense (layers 0-2, 18,432) or MoE (256 experts of 2,048, top-8 by a sigmoid gate with a bias -
//!                  within the best 4 of 8 groups of 32 - the weights normalized, x2.5, plus one shared expert)
//! ```
//!
//! The file's attention keeps kv_b whole (`attn_kv_b`, [128 x (128 + 128), 512]: per head the k part, then the v
//! part); newer conversions split it (`attn_k_b`, `attn_v_b`). The engine splits it at load.

use nextsycl_gguf::{GType, Gguf, Tensor, Value};

use crate::{Error, ModelResult as Result};

/// The geometry, from metadata and the tensors' shapes.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub n_embd: u64,
    pub n_vocab: u64,
    pub n_layer: u64,
    pub n_dense: u64,
    pub ffn_dense: u64,
    pub n_expert: u64,
    pub n_expert_used: u64,
    pub n_expert_shared: u64,
    pub ffn_expert: u64,
    pub expert_scale: f64,
    pub expert_norm: bool,
    /// 2 = sigmoid (V3 / R1), 1 = softmax (V2)
    pub gating: u64,
    pub rms_eps: f64,
    pub n_head: u64,
    pub q_lora: u64,
    pub kv_lora: u64,
    /// the key's width a head (nope + rope: 192) and the value's (128)
    pub key_len: u64,
    pub value_len: u64,
    /// the rotated part (64)
    pub rope_dim: u64,
    pub rope_base: f64,
    /// YaRN: the context factor, the training context, the attention scale's log multiplier (as the file stores it)
    pub yarn_factor: f64,
    pub yarn_orig_ctx: u64,
    pub yarn_log_mul: f64,
}

impl Geometry {
    pub fn is_moe(&self, layer: u64) -> bool {
        layer >= self.n_dense
    }
    /// the key's part without the rotation (128)
    pub fn nope(&self) -> u64 {
        self.key_len - self.rope_dim
    }
    /// a cache row's values: the latent and the rotated key part (576)
    pub fn cache_row(&self) -> u64 {
        self.kv_lora + self.rope_dim
    }
}

/// A tensor's role. `L` = in a layer, the rest global.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    TokenEmbd,
    OutputNorm,
    Output,
    AttnNorm,
    FfnNorm,
    QA,
    QANorm,
    QB,
    KvA,
    KvANorm,
    /// the whole kv_b (legacy files): [n_head * (nope + value), kv_lora]
    KvB,
    /// the split ones (newer files): [n_head, kv_lora, nope] and [n_head, value, kv_lora]
    KB,
    VB,
    Out,
    FfnGate,
    FfnUp,
    FfnDown,
    Router,
    RouterBias,
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
    Edges,
    Attention,
    Dense,
    /// the router and the shared expert: every token, always resident
    Shared,
    /// the routed experts: 8 of 256 a token, the rest of VRAM, pinned memory and the file
    Experts,
}

impl Role {
    pub fn group(self) -> Group {
        use Role::*;
        match self {
            TokenEmbd | OutputNorm | Output => Group::Edges,
            FfnGate | FfnUp | FfnDown => Group::Dense,
            Router | RouterBias | ShGate | ShUp | ShDown => Group::Shared,
            ExpGate | ExpUp | ExpDown => Group::Experts,
            _ => Group::Attention,
        }
    }

    /// The tensor's name suffix (after `blk.N.`; the whole name for a global role).
    pub fn name(self) -> &'static str {
        use Role::*;
        match self {
            TokenEmbd => "token_embd.weight",
            OutputNorm => "output_norm.weight",
            Output => "output.weight",
            AttnNorm => "attn_norm.weight",
            FfnNorm => "ffn_norm.weight",
            QA => "attn_q_a.weight",
            QANorm => "attn_q_a_norm.weight",
            QB => "attn_q_b.weight",
            KvA => "attn_kv_a_mqa.weight",
            KvANorm => "attn_kv_a_norm.weight",
            KvB => "attn_kv_b.weight",
            KB => "attn_k_b.weight",
            VB => "attn_v_b.weight",
            Out => "attn_output.weight",
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
        }
    }

    fn global(self) -> bool {
        matches!(self, Role::TokenEmbd | Role::OutputNorm | Role::Output)
    }
}

const COMMON: [Role; 9] = [Role::AttnNorm, Role::FfnNorm, Role::QA, Role::QANorm, Role::QB, Role::KvA, Role::KvANorm, Role::Out, Role::KvB];
const DENSE: [Role; 3] = [Role::FfnGate, Role::FfnUp, Role::FfnDown];
const MOE: [Role; 8] = [Role::Router, Role::RouterBias, Role::ExpGate, Role::ExpUp, Role::ExpDown, Role::ShGate, Role::ShUp, Role::ShDown];

/// A model file, resolved: its geometry and every tensor by (layer, role).
pub struct Model<'g> {
    pub file: &'g Gguf,
    pub g: Geometry,
    /// kv_b split in the file (attn_k_b / attn_v_b) rather than whole
    pub split_kv: bool,
}

fn meta_u(f: &Gguf, keys: &[&str]) -> Result<u64> {
    keys.iter().find_map(|k| f.arch_meta(k).and_then(Value::as_u64)).ok_or_else(|| Error(format!("metadata {} is missing", keys[0])))
}
fn meta_f(f: &Gguf, keys: &[&str], default: Option<f64>) -> Result<f64> {
    keys.iter()
        .find_map(|k| f.arch_meta(k).and_then(|v| v.as_f64()))
        .or(default)
        .ok_or_else(|| Error(format!("metadata {} is missing", keys[0])))
}

impl<'g> Model<'g> {
    pub fn open(file: &'g Gguf) -> Result<Model<'g>> {
        let split_kv = file.tensor("blk.0.attn_k_b.weight").is_some();
        let n_vocab = meta_u(file, &["vocab_size"]).or_else(|_| file.tensor(Role::TokenEmbd.name()).map(|t| t.shape[0]).ok_or(Error("token_embd is missing".into())))?;
        let g = Geometry {
            n_embd: meta_u(file, &["embedding_length"])?,
            n_vocab,
            n_layer: meta_u(file, &["block_count"])?,
            n_dense: meta_u(file, &["leading_dense_block_count"])?,
            ffn_dense: meta_u(file, &["feed_forward_length"])?,
            n_expert: meta_u(file, &["expert_count"])?,
            n_expert_used: meta_u(file, &["expert_used_count"])?,
            n_expert_shared: meta_u(file, &["expert_shared_count"])?,
            ffn_expert: meta_u(file, &["expert_feed_forward_length"])?,
            expert_scale: meta_f(file, &["expert_weights_scale"], Some(1.0))?,
            expert_norm: meta_u(file, &["expert_weights_norm"]).unwrap_or(0) != 0,
            gating: meta_u(file, &["expert_gating_func"]).unwrap_or(1),
            rms_eps: meta_f(file, &["attention.layer_norm_rms_epsilon"], None)?,
            n_head: meta_u(file, &["attention.head_count"])?,
            q_lora: meta_u(file, &["attention.q_lora_rank"])?,
            kv_lora: meta_u(file, &["attention.kv_lora_rank"])?,
            key_len: meta_u(file, &["attention.key_length_mla", "attention.key_length"])?,
            value_len: meta_u(file, &["attention.value_length_mla", "attention.value_length"])?,
            rope_dim: meta_u(file, &["rope.dimension_count"])?,
            rope_base: meta_f(file, &["rope.freq_base"], Some(10000.0))?,
            yarn_factor: meta_f(file, &["rope.scaling.factor"], Some(1.0))?,
            yarn_orig_ctx: meta_u(file, &["rope.scaling.original_context_length"]).unwrap_or(4096),
            yarn_log_mul: meta_f(file, &["rope.scaling.yarn_log_multiplier"], Some(0.0))?,
        };
        if g.gating != 2 {
            return Err(Error(format!("expert gating {} (this engine runs DeepSeek-V3 / R1's sigmoid gate, 2)", g.gating)));
        }
        Ok(Model { file, g, split_kv })
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
        let mut v: Vec<Role> = COMMON.to_vec();
        if self.split_kv {
            v.retain(|r| *r != Role::KvB);
            v.extend([Role::KB, Role::VB]);
        }
        if self.g.is_moe(layer) {
            v.extend(MOE);
        } else {
            v.extend(DENSE);
        }
        v
    }

    /// The shape a role must have (row-major, outermost first).
    pub fn shape(&self, r: Role) -> Vec<u64> {
        use Role::*;
        let g = &self.g;
        let d = g.n_embd;
        match r {
            TokenEmbd | Output => vec![g.n_vocab, d],
            OutputNorm | AttnNorm | FfnNorm => vec![d],
            QA => vec![g.q_lora, d],
            QANorm => vec![g.q_lora],
            QB => vec![g.n_head * g.key_len, g.q_lora],
            KvA => vec![g.kv_lora + g.rope_dim, d],
            KvANorm => vec![g.kv_lora],
            KvB => vec![g.n_head * (g.nope() + g.value_len), g.kv_lora],
            KB => vec![g.n_head, g.kv_lora, g.nope()],
            VB => vec![g.n_head, g.value_len, g.kv_lora],
            Out => vec![d, g.n_head * g.value_len],
            FfnGate | FfnUp => vec![g.ffn_dense, d],
            FfnDown => vec![d, g.ffn_dense],
            Router => vec![g.n_expert, d],
            RouterBias => vec![g.n_expert],
            ExpGate | ExpUp => vec![g.n_expert, g.ffn_expert, d],
            ExpDown => vec![g.n_expert, d, g.ffn_expert],
            ShGate | ShUp => vec![g.ffn_expert * g.n_expert_shared, d],
            ShDown => vec![d, g.ffn_expert * g.n_expert_shared],
        }
    }

    /// Every block's every role: present and of the right shape; no tensor in the file left over. All problems.
    pub fn check(&self) -> Vec<String> {
        let mut errs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for r in [Role::TokenEmbd, Role::OutputNorm, Role::Output] {
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

    /// YaRN as llama.cpp runs this architecture (llama-context.cpp, models/deepseek2.cpp): (freq_scale, ext_factor,
    /// the cos / sin factor, corr0, corr1, the attention scale). The file's log multiplier was written x 0.1 by the
    /// converter (llama.cpp's TAG_DEEPSEEK2_YARN_LOG_MUL_FIX); the rotation's own magnitude factor cancels.
    pub fn yarn(&self) -> (f32, f32, f32, f32, f32, f32) {
        let g = &self.g;
        let factor = g.yarn_factor as f32;
        let yarn = factor > 1.0;
        let freq_scale = if yarn { 1.0f32 / factor } else { 1.0 };
        let ext_factor = if yarn { 1.0f32 } else { 0.0 };
        let log_mul = g.yarn_log_mul as f32 / 0.1;
        let get_mscale = |s: f32, m: f32| if s <= 1.0 { 1.0f32 } else { 0.1 * m * s.ln() + 1.0 };
        // the context's attention factor: mscale (1) over mscale_all_dims (log_mul), then the rotation's own
        // 1 + 0.1 ln(1 / freq_scale) cancelled
        let mut attn_factor = if yarn && log_mul != 0.0 { get_mscale(1.0 / freq_scale, 1.0) / get_mscale(1.0 / freq_scale, log_mul) } else { get_mscale(1.0 / freq_scale, 1.0) };
        if yarn {
            attn_factor *= 1.0 / (1.0 + 0.1 * factor.ln());
        }
        // ggml's rope_yarn: mscale = attn_factor, then x (1 + 0.1 ln(1 / freq_scale)) when ext_factor != 0
        let rope_mscale = if ext_factor != 0.0 { attn_factor * (1.0 + 0.1 * (1.0 / freq_scale).ln()) } else { attn_factor };
        // the scores' scale: (the original attention factor x (1 + 0.1 log_mul ln(1 / freq_scale)))^2 / sqrt(192)
        let attn_org = attn_factor * (1.0 + 0.1 * (1.0 / freq_scale).ln());
        let mscale = attn_org * (1.0 + 0.1 * log_mul * (1.0 / freq_scale).ln());
        let kq_scale = mscale * mscale / (g.key_len as f32).sqrt();
        // ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, base, beta_fast 32, beta_slow 1)
        let n = g.rope_dim as f32;
        let corr = |beta: f32| n * (g.yarn_orig_ctx as f32 / (beta * 2.0 * std::f32::consts::PI)).ln() / (2.0 * (g.rope_base as f32).ln());
        let c0 = corr(32.0).floor().max(0.0);
        let c1 = corr(1.0).ceil().min(n - 1.0);
        (freq_scale, ext_factor, rope_mscale, c0, c1, kq_scale)
    }
}
