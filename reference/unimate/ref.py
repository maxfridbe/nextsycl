"""UniMate's reference dumps, stage by stage, for `nextsycl model3d check` (model3d/unimate/src/check.rs).

Runs the official modules of github.com/Friedrich-M/UniMate @ b78c780 (MIT; the checkpoints are CC BY-NC 4.0) in float32
on the CPU, transcribed here (nothing of the repository, nor of a checkpoint, is imported or executed):
- unimate/models/denoiser/{base,graph_adaln,rope}.py and blocks/{common,graph}.py (the `graph` x `adaln` variant),
- unimate/models/flow/{transport,path,integrators}.py (the linear path, velocity model, torchdiffeq's dopri5),
- unimate/inference/generate.py (classifier-free guidance), unimate/dataset/{conditioning,transforms}.py (the
  conditioning a skeleton gives the model), unimate/utils/{topology_utils,motion_utils,rotation_conversions}.py,
- the text encoder: transformers' T5EncoderModel on google/flan-t5-base, mean-pooled as unimate/models/text_encoder/t5.py.
The checkpoint is read by safeio.py (never torch.load) and the EMA weights are the ones used, as sample.py does.

    python ref.py --unimate DIR --t5 DIR --cond COND.npy [--name KEY] --prompt "An object walks forward." \
        --out DIR [--seed 10] [--cfg 3]

DIR/unimate holds config.json, dataset_stats.npy, checkpoints/checkpoint_step_150000.pt; COND.npy is a rig_preprocess
cond (read by safeio's restricted unpickler). Dumps (float32 .npy; integers as float32 unless said):
    meta.json                 n joints, prompt, seed, cfg, the ode's evaluations and accepted steps
    tokens.npy [T]            the prompt's T5 ids (int64)
    t5.npy [T, 768]           T5's last hidden state for the prompt
    caption.npy [768]         its mean (the adaLN vector's input)
    names_emb.npy [J, 768]    the pooled T5 embedding of each cleaned joint name
    in_*.npy                  the model's conditioning, padded to 71 joints: tpos [71,12] (normalized), tpos_parents,
                              depths [71], spectral [71,8], graph_dist [71,71], relations [71,71], mean, std [71,12]
    noise.npy [71,12,60]      the flow's starting noise
    pass_*.npy                one model pass at t = 0.37 on the noise: y [512] (the adaLN vector), tokens0 [61,71,512]
                              (after the input layer and embeddings), block0, block9 [61,71,512], cond / uncond / guided
                              [71,12,60]
    ode.npy [71,12,60]        the dopri5 solution at t = 1 (atol 1e-6, rtol 1e-3, 50 saved times, guidance cfg)
    motion.npy [60,J,12]      denormalized features of the first J joints
    quats.npy [60,J,4]        the decoded local rotations (w x y z, w >= 0), root_pos.npy [60,3] the root's path
    param_order.txt           name and shape of each of the model's parameters, in parameters() order
"""
import argparse
import json
import math
import os
import sys

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safeio  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# blocks/common.py
# ---------------------------------------------------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features, bias=True):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=torch.float32):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half).to(device=t.device, dtype=dtype)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t, dtype=torch.float32):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size, dtype=dtype))


class TposCrossAttentionPool(nn.Module):
    def __init__(self, latent_dim, num_queries=4, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, latent_dim) * 0.02)
        self.norm_q = LlamaRMSNorm(latent_dim, eps=1e-6)
        self.norm_kv = LlamaRMSNorm(latent_dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.norm_ff = LlamaRMSNorm(latent_dim, eps=1e-6)
        self.ffn = SwiGLUFFN(latent_dim, int(latent_dim * mlp_ratio))
        self.norm_out = LlamaRMSNorm(latent_dim, eps=1e-6)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, tpos_emb, joints_valid):
        B = tpos_emb.shape[0]
        q = self.queries.expand(B, -1, -1)
        key_padding_mask = ~joints_valid
        kv = self.norm_kv(tpos_emb)
        attn_out, _ = self.cross_attn(self.norm_q(q), kv, kv, key_padding_mask=key_padding_mask)
        q = q + attn_out
        q = q + self.ffn(self.norm_ff(q))
        pooled = q.mean(dim=1)
        return self.out_proj(self.norm_out(pooled))


class TposPool(nn.Module):
    def __init__(self, latent_dim, num_queries=0):
        super().__init__()
        self.num_queries = num_queries
        self.pool = TposCrossAttentionPool(latent_dim, num_queries=num_queries)

    def forward(self, tpos_emb, joints_valid):
        return self.pool(tpos_emb, joints_valid)


class InputLayer(nn.Module):
    def __init__(self, feature_len, latent_dim, max_joints, concat_parent_features=True):
        super().__init__()
        self.max_joints = max_joints
        self.concat_parent_features = concat_parent_features

        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, o), nn.SiLU(), nn.Linear(o, o))
        self.root_tpos_embedder = mlp(feature_len, latent_dim)
        self.root_x_embedder = mlp(feature_len, latent_dim)
        self.joint_tpos_embedder = mlp(feature_len, latent_dim)
        self.parent_tpos_embedder = mlp(feature_len, latent_dim)
        self.tpos_fuse = mlp(latent_dim * 2, latent_dim)
        self.joint_x_embedder = mlp(feature_len, latent_dim)

    def forward(self, x, cond):
        tpos = einops.rearrange(cond['tpos_first_frame'], 'b j d -> b 1 j d')
        tpos_root = self.root_tpos_embedder(tpos[:, :, :1, :])
        tpos_joint_proj = self.joint_tpos_embedder(tpos[:, :, 1:, :])
        tpos_parents = einops.rearrange(cond['tpos_first_frame_parents'], 'b j d -> b 1 j d')
        tpos_parent_proj = self.parent_tpos_embedder(tpos_parents[:, :, 1:, :])
        tpos_joints = self.tpos_fuse(torch.cat([tpos_joint_proj, tpos_parent_proj], dim=-1))
        tpos_emb = torch.cat([tpos_root, tpos_joints], dim=2)
        joints_valid = torch.arange(self.max_joints).expand(len(cond['n_joints']), self.max_joints) < cond['n_joints'].unsqueeze(1)
        x = einops.rearrange(x, 'b j d f -> b f j d')
        x_root = self.root_x_embedder(x[:, :, :1, :])
        x_joints = self.joint_x_embedder(x[:, :, 1:, :])
        x = torch.cat([x_root, x_joints], dim=2)
        x = torch.cat([tpos_emb, x], dim=1)
        return x, tpos_emb[:, 0], joints_valid


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size, joint=24, num_heads=4):
        super().__init__()
        self.norm_final = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.joint = joint
        self.root_cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.root_cross_norm_q = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.root_cross_norm_kv = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.root_out = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, output_size))
        self.joint_out = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, output_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c, joints_valid=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        B, Fm, J, D = x.shape
        x = x.reshape(B, Fm * J, D)
        norm_x = self.norm_final(x.to(torch.float32)).to(x.dtype)
        x = modulate(norm_x, shift[:, None, :], scale[:, None, :])
        x = einops.rearrange(x, 'b (f j) d -> b f j d', j=self.joint)
        B, Fm, J, D = x.shape
        root_tokens = x[:, :, :1, :]
        joint_tokens = x[:, :, 1:, :]
        root_flat = root_tokens.reshape(B * Fm, 1, D)
        joints_flat = joint_tokens.reshape(B * Fm, J - 1, D)
        kv_mask = ~joints_valid[:, 1:]
        kv_mask = kv_mask.unsqueeze(1).expand(-1, Fm, -1).reshape(B * Fm, J - 1)
        root_agg, _ = self.root_cross_attn(self.root_cross_norm_q(root_flat), self.root_cross_norm_kv(joints_flat), joints_flat,
                                           key_padding_mask=kv_mask)
        root_tokens = root_tokens + root_agg.reshape(B, Fm, 1, D)
        root = self.root_out(root_tokens)
        joints = self.joint_out(joint_tokens)
        x = torch.cat([root, joints], dim=2)
        return einops.rearrange(x, 'b f j d -> b d f j')


# ---------------------------------------------------------------------------------------------------------------------
# rope.py
# ---------------------------------------------------------------------------------------------------------------------
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    dtype = q.dtype
    q_f, k_f = q.float(), k.float()
    return ((q_f * cos) + (_rotate_half(q_f) * sin)).to(dtype), ((k_f * cos) + (_rotate_half(k_f) * sin)).to(dtype)


class RopeND:
    def __init__(self, head_dim=64, nd=1, max_lens=(1024,), nd_split=(1,), cache_longer=1):
        self.nd, self.head_dim, self.max_lens, self.nd_split, self.cache_longer = nd, head_dim, list(max_lens), list(nd_split), cache_longer
        self.split_dims = [2 * s * (head_dim // 2 // sum(nd_split)) for s in nd_split]
        self.bases = [(int(8 * L / math.pi) // 100 + 1) * 100 for L in max_lens]

    def _generate_cos_sin(self, max_len, dim, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len * self.cache_longer).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=1)
        return freqs.cos().float(), freqs.sin().float()

    def __call__(self, q, k, position_ids):
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        c, s = self._generate_cos_sin(self.max_lens[0], self.split_dims[0], self.bases[0])
        return _apply_rotary_pos_emb(q, k, c[position_ids[0, :], :][None, None], s[position_ids[0, :], :][None, None])


class SignNetSpectralEncoder(nn.Module):
    def __init__(self, num_eigvecs, hidden_dim, out_dim):
        super().__init__()
        self.num_eigvecs = num_eigvecs
        self.phi = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.rho = nn.Sequential(nn.Linear(num_eigvecs * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, spectral_coords):
        v = spectral_coords.unsqueeze(-1)
        h = self.phi(v) + self.phi(-v)
        return self.rho(h.flatten(start_dim=-2))


class SpectralJointRoPE(nn.Module):
    def __init__(self, head_dim=64, num_eigvecs=8, use_signnet=True, signnet_hidden=64):
        super().__init__()
        self.head_dim = head_dim
        self.spectral_encoder = SignNetSpectralEncoder(num_eigvecs, signnet_hidden, head_dim // 2)

    def _compute_cos_sin(self, spectral_coords):
        angles = self.spectral_encoder(spectral_coords)
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos(), angles.sin()

    def forward_per_frame(self, q, k, spectral_coords, n_frames):
        B, J = spectral_coords.shape[0], spectral_coords.shape[1]
        H, C = q.shape[1], q.shape[3]
        cos, sin = self._compute_cos_sin(spectral_coords)
        cos = cos.view(B, 1, 1, J, C)
        sin = sin.view(B, 1, 1, J, C)
        q = q.reshape(B, n_frames, H, J, C)
        k = k.reshape(B, n_frames, H, J, C)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        return q.reshape(B * n_frames, H, J, C), k.reshape(B * n_frames, H, J, C)


# ---------------------------------------------------------------------------------------------------------------------
# blocks/graph.py
# ---------------------------------------------------------------------------------------------------------------------
def build_graph_bias_params(module, dim, num_heads, max_path_len=5, edge_types=6):
    module.graph_dist_embedding = nn.Embedding(max_path_len + 1, dim // 4)
    module.graph_dist_proj = nn.Linear(dim // 4, num_heads)
    module.graph_dist_scale = nn.Parameter(torch.ones(1) * 0.02)
    module.graph_rel_embedding = nn.Embedding(edge_types, dim // 4)
    module.graph_rel_proj = nn.Linear(dim // 4, num_heads)
    module.graph_rel_scale = nn.Parameter(torch.ones(1) * 0.02)


def compute_graph_bias(module, cond, attention_mask):
    dist_emb = module.graph_dist_proj(module.graph_dist_embedding(cond["graph_dist"])) * module.graph_dist_scale
    rel_emb = module.graph_rel_proj(module.graph_rel_embedding(cond["joint_relations"])) * module.graph_rel_scale
    graph_bias = (dist_emb + rel_emb).permute(0, 3, 1, 2)
    return graph_bias.masked_fill(~attention_mask.to(torch.bool), float('-inf'))


class SpatialGraphSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, rope=None, qk_norm=True, max_path_len=5, edge_types=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = rope
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)
        build_graph_bias_params(self, dim, num_heads, max_path_len, edge_types)

    def forward(self, x, attention_mask=None, cond=None):
        B, Fr, J, _ = x.shape
        C = self.num_heads * self.head_dim
        x_flat = einops.rearrange(x, 'b f j d -> (b f) j d')
        qkv = self.qkv(x_flat).reshape(B * Fr, J, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope.forward_per_frame(q, k, cond['spectral_feats'], n_frames=Fr)
        graph_bias = compute_graph_bias(self, cond, attention_mask)
        q = q.reshape(B, Fr, self.num_heads, J, self.head_dim)
        k = k.reshape(B, Fr, self.num_heads, J, self.head_dim)
        v = v.reshape(B, Fr, self.num_heads, J, self.head_dim)
        attn_weights = (q @ k.transpose(-1, -2)) * self.scale
        attn_weights = attn_weights + graph_bias.unsqueeze(1)
        attn_weights = attn_weights.float().softmax(dim=-1).to(v.dtype)
        out = (attn_weights @ v).reshape(B * Fr, self.num_heads, J, self.head_dim)
        out = out.transpose(1, 2).reshape(B * Fr, J, C)
        out = self.proj(out)
        return einops.rearrange(out, '(b f) j d -> b f j d', b=B, f=Fr)


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, rope=None, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x, attention_mask=None, position_ids=None):
        B_orig, Fr, J, _ = x.shape
        x = einops.rearrange(x, 'b f j d -> (b j) f d')
        attention_mask = einops.repeat(attention_mask, 'b 1 1 f -> (b j) 1 1 f', j=J).to(torch.bool)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, k, position_ids)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return einops.rearrange(x, '(b j) f d -> b f j d', b=B_orig, j=J)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size=1024, rope_spatial=None, rope_temporal=None):
        super().__init__()
        self.norm_s = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.s_attn = SpatialGraphSelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, rope=rope_spatial)
        self.norm_t = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.t_attn = TemporalSelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, rope=rope_temporal)
        self.norm_mlp = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))

    def forward(self, x, c, spatial_mask=None, temporal_mask=None, position_ids_temporal=None, cond=None):
        dtype = x.dtype
        (shift_s, scale_s, gate_s, shift_t, scale_t, gate_t, shift_m, scale_m, gate_m) = self.adaLN_modulation(c).chunk(9, dim=-1)

        def e(z):
            return z[:, None, None, :]
        xs = self.norm_s(x.to(torch.float32)).to(dtype)
        xs = modulate(xs, e(shift_s), e(scale_s))
        xs = self.s_attn(xs, attention_mask=spatial_mask, cond=cond)
        x = x + e(gate_s) * xs
        xt = self.norm_t(x.to(torch.float32)).to(dtype)
        xt = modulate(xt, e(shift_t), e(scale_t))
        xt = self.t_attn(xt, attention_mask=temporal_mask, position_ids=position_ids_temporal)
        x = x + e(gate_t) * xt
        xm = self.norm_mlp(x.to(torch.float32)).to(dtype)
        xm = modulate(xm, e(shift_m), e(scale_m))
        x = x + e(gate_m) * self.mlp(xm)
        return x


# ---------------------------------------------------------------------------------------------------------------------
# denoiser/base.py + graph_adaln.py
# ---------------------------------------------------------------------------------------------------------------------
class UniMateGraphAdaLN(nn.Module):
    def __init__(self, feature_len=12, max_motion_length=60, max_joints=71, max_depth=20, latent_dim=512, ff_size=2048,
                 num_layers=10, num_heads=8, max_freqs=8, text_dim=768, num_tpos_queries=4):
        super().__init__()
        self.max_motion_length, self.max_joints, self.latent_dim, self.text_dim = max_motion_length, max_joints, latent_dim, text_dim
        self.n_prefix = 1
        self.cond_embedder = nn.Linear(text_dim, latent_dim)
        self.time_embedder = TimestepEmbedder(hidden_size=latent_dim)
        self.input_layer = InputLayer(feature_len, latent_dim, max_joints, True)
        self.tpos_pool = TposPool(latent_dim, num_queries=num_tpos_queries)
        self.joint_name_embedder = nn.Linear(text_dim, latent_dim)
        self.joint_name_dropout = nn.Dropout(0.0)
        self.max_depth = max_depth
        self.depth_embedding = nn.Embedding(max_depth + 1, latent_dim)
        head_dim = latent_dim // num_heads
        max_frames = max_motion_length + self.n_prefix
        self.rope_j = SpectralJointRoPE(head_dim=head_dim, num_eigvecs=max_freqs)
        self.rope_t = RopeND(head_dim=head_dim, nd=1, nd_split=[1], max_lens=[max_frames])
        self.temporal_position_ids = torch.arange(max_frames).unsqueeze(0)
        self.transformer_blocks = nn.ModuleList([SpatioTemporalBlock(latent_dim, num_heads, ff_size, self.rope_j, self.rope_t)
                                                 for _ in range(num_layers)])
        self.final_layer = FinalLayer(latent_dim, feature_len, joint=max_joints)
        self.trace = None

    def forward(self, x, timesteps, cond=None, force_mask=False):
        bs, njoints, nfeats, nframes = x.shape
        timesteps_emb = self.time_embedder(timesteps, dtype=x.dtype)
        cv = cond['caption_emb']
        cv = torch.zeros_like(cv) if force_mask else cv
        y = timesteps_emb + self.cond_embedder(cv)
        x, tpos_emb, joints_valid = self.input_layer(x, cond)
        y = y + self.tpos_pool(tpos_emb, joints_valid)
        joint_depths = cond['joint_depths'].long().clamp(max=self.max_depth)
        x = x + self.depth_embedding(joint_depths).unsqueeze(1)
        x = x + self.joint_name_dropout(self.joint_name_embedder(cond['joint_names_emb'])).unsqueeze(1)
        if self.trace is not None:
            self.trace['y'], self.trace['tokens0'] = y.detach().clone(), x.detach().clone()
        temporal_valid = torch.arange(nframes + self.n_prefix).expand(len(cond['motion_length']), nframes + self.n_prefix) < (cond['motion_length'] + self.n_prefix).unsqueeze(1)
        temporal_mask = temporal_valid.unsqueeze(1).unsqueeze(1)
        joint_mask = joints_valid.unsqueeze(1).unsqueeze(2)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, y, spatial_mask=joint_mask, temporal_mask=temporal_mask,
                      position_ids_temporal=self.temporal_position_ids[:, :nframes + self.n_prefix], cond=cond)
            if self.trace is not None and i in (0, len(self.transformer_blocks) - 1):
                self.trace[f'block{i}'] = x.detach().clone()
        x = self.final_layer(x, y, joints_valid=joints_valid)
        x = einops.rearrange(x, 'b d f j -> b j d f')
        return x[:, :, :, self.n_prefix:]


# ---------------------------------------------------------------------------------------------------------------------
# utils/topology_utils.py
# ---------------------------------------------------------------------------------------------------------------------
def children_map(parents):
    n = len(parents)
    ch = [[] for _ in range(n)]
    for c, p in enumerate(parents):
        if p != -1:
            ch[p].append(c)
    return ch


def compute_laplacian_eigenvectors(parents, max_freqs=8):
    parents = np.asarray(parents, dtype=np.int64)
    n = len(parents)
    k = min(n - 1, max_freqs)
    A = np.zeros((n, n))
    for c, p in enumerate(parents):
        if p >= 0:
            A[p, c] = A[c, p] = 1.0
    deg = A.sum(1)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    L = np.eye(n) - dinv[:, None] * A * dinv[None, :]
    w, v = np.linalg.eigh(L)
    vecs = v[:, 1:1 + k]
    for j in range(vecs.shape[1]):
        d = np.sqrt((vecs[:, j] ** 2).sum())
        if d > 1e-12:
            vecs[:, j] = vecs[:, j] / d
    if k < max_freqs:
        vecs = np.pad(vecs, ((0, 0), (0, max_freqs - k)))
    return vecs.astype(np.float32)


def compute_edge_relations_and_distances(parents, max_path_len=5):
    from collections import deque
    parents = list(parents)
    n = len(parents)
    children = children_map(parents)
    rel = np.full((n, n), 4, dtype=np.int16)
    for i in range(n):
        pi = parents[i]
        for j in range(n):
            pj = parents[j]
            if i == j:
                rel[i, j] = 0
            elif pj == i:
                rel[i, j] = 2
            elif j == pi and pi != -1:
                rel[i, j] = 1
            elif pi != -1 and pj == pi:
                rel[i, j] = 3
        if len(children[i]) == 0:
            rel[i, i] = 5
    adj = [[] for _ in range(n)]
    for c, p in enumerate(parents):
        if p != -1:
            adj[c].append(p)
            adj[p].append(c)
    topo = np.full((n, n), max_path_len, dtype=np.int16)
    for s in range(n):
        dist = np.full(n, 32767, dtype=np.int16)
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= max_path_len:
                continue
            for v in adj[u]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        topo[s] = np.minimum(dist, max_path_len)
    return rel, topo


def compute_joint_depths(parents):
    n = len(parents)
    d = np.full(n, -1, dtype=np.int64)
    for j in range(n):
        if parents[j] == -1 or parents[j] == j:
            d[j] = 0
    changed = True
    while changed:
        changed = False
        for j in range(n):
            if d[j] >= 0:
                continue
            p = parents[j]
            if 0 <= p < n and d[p] >= 0:
                d[j] = d[p] + 1
                changed = True
    return d


# ---------------------------------------------------------------------------------------------------------------------
# utils/rotation_conversions.py + motion_utils.py (the parts the sampler's output goes through)
# ---------------------------------------------------------------------------------------------------------------------
IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _gram_schmidt_np(c):
    x_raw, y_raw = c[..., 0:3], c[..., 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw, axis=-1)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)


def rotation_6d_to_matrix_np(c):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        mats = _gram_schmidt_np(c)
        nx = np.linalg.norm(c[..., 0:3], axis=-1, keepdims=True)
        ny = np.linalg.norm(c[..., 3:6], axis=-1, keepdims=True)
        sine = np.linalg.norm(np.cross(c[..., 0:3] / nx, c[..., 3:6] / ny, axis=-1), axis=-1)
        valid = (np.isfinite(c).all(-1) & np.isfinite(mats).reshape(*mats.shape[:-2], 9).all(-1)
                 & (nx[..., 0] > 1e-6) & (ny[..., 0] > 1e-6) & (sine > 1e-6))
    if valid.all():
        return mats
    return _gram_schmidt_np(np.where(valid[..., None], c, np.asarray(IDENTITY_6D, dtype=mats.dtype)))


def matrix_to_quaternion_np(mats):
    m = np.asarray(mats, dtype=np.float64)
    d0, d1, d2 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    cand = np.stack([1 + d0 + d1 + d2, 1 + d0 - d1 - d2, 1 - d0 + d1 - d2, 1 - d0 - d1 + d2], axis=-1)
    branch = np.argmax(cand, axis=-1)
    s = 2.0 * np.sqrt(np.maximum(np.take_along_axis(cand, branch[..., None], -1)[..., 0], 1e-12))
    m21_m12, m02_m20, m10_m01 = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
    m21p12, m02p20, m10p01 = m[..., 2, 1] + m[..., 1, 2], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 0] + m[..., 0, 1]
    q = np.where(branch[..., None] == 0, np.stack([s / 4, m21_m12 / s, m02_m20 / s, m10_m01 / s], -1),
        np.where(branch[..., None] == 1, np.stack([m21_m12 / s, s / 4, m10p01 / s, m02p20 / s], -1),
        np.where(branch[..., None] == 2, np.stack([m02_m20 / s, m10p01 / s, s / 4, m21p12 / s], -1),
                 np.stack([m10_m01 / s, m02p20 / s, m21p12 / s, s / 4], -1))))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def hml_rotations_to_bvh_quaternions(rot6d, parents):
    T, J, _ = rot6d.shape
    bvh = np.tile(np.eye(3), (T, J, 1, 1))
    hml = rotation_6d_to_matrix_np(rot6d)
    copy_sum = np.zeros((T, J, 3, 3))
    has_child = np.zeros(J, dtype=bool)
    for j, p in enumerate(parents[1:], 1):
        copy_sum[:, p] += hml[:, j]
        has_child[p] = True
    U, _, Vt = np.linalg.svd(copy_sum[:, has_child])
    U[..., :, -1] *= np.sign(np.linalg.det(U @ Vt))[..., None]
    bvh[:, has_child] = U @ Vt
    return matrix_to_quaternion_np(bvh)


def recover_root(data):
    """data: root features (T, 12) -> (facing matrices (T,3,3), root positions (T,3))"""
    R = rotation_6d_to_matrix_np(data[:, 3:9])
    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[1:, [0, 2]] = data[:-1, [9, 11]]
    r_pos = np.einsum('tji,tj->ti', R, r_pos)       # -q * v: the inverse rotation, R^T v
    r_pos = np.cumsum(r_pos, axis=-2)
    r_pos[:, 1] = data[:, 1]
    return R, r_pos


# ---------------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unimate", required=True, help="the run directory (config.json, dataset_stats.npy, checkpoints/)")
    ap.add_argument("--t5", required=True, help="google/flan-t5-base's folder")
    ap.add_argument("--cond", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--prompt", default="An object walks forward.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--ckpt", default="checkpoints/checkpoint_step_150000.pt")
    ap.add_argument("--no-ode", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    cfgj = json.load(open(os.path.join(a.unimate, "config.json")))
    mc, dc = cfgj["model"], cfgj["dataset"]
    MAXJ, T, D = dc["max_joints"], dc["max_motion_length"], dc["feature_len"]

    def save(name, x):
        np.save(os.path.join(a.out, name + ".npy"), np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float32 if np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x).dtype.kind == 'f' else np.int64))

    # ---- the text encoder
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(a.t5)
    t5 = T5EncoderModel.from_pretrained(a.t5, torch_dtype=torch.float32).eval()

    def encode(texts):
        enc = tok(texts, return_tensors="pt", padding=True)
        empty = torch.LongTensor([i for i, e in enumerate(texts) if e == ""])
        enc["attention_mask"][empty, :] = 0
        hidden = t5(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        n = enc["attention_mask"].sum(-1)
        return enc["input_ids"], [hidden[i, :max(int(n[i]), 1)].numpy() if n[i] > 0 else np.zeros((1, hidden.shape[-1]), np.float32) for i in range(len(texts))]

    ids, [prompt_tokens] = encode([a.prompt])
    save("tokens", ids[0].numpy())
    save("t5", prompt_tokens)
    caption_emb = prompt_tokens.mean(axis=0)
    save("caption", caption_emb)

    # ---- the skeleton's conditioning
    conds = safeio.load_npy_dict(a.cond)
    key = a.name or list(conds.keys())[0]
    ce = conds[key]
    parents = np.asarray(ce["parents"], dtype=np.int64)
    J = len(parents)
    names = [str(n) for n in ce["clean_joint_names"]]
    uniq = sorted(set(names))
    _, embs = encode(uniq)
    by_name = {n: e.mean(axis=0) for n, e in zip(uniq, embs)}
    names_emb = np.stack([by_name[n] for n in names]).astype(np.float32)
    save("names_emb", names_emb)

    stats = safeio.load_npy_dict(os.path.join(a.unimate, "dataset_stats.npy"))
    st = stats[list(stats.keys())[0]]
    tpos = np.asarray(ce["tpos_first_frame"], dtype=np.float64).copy()
    tpos[0, [0, 2]] = 0.0
    tpos[..., 1] -= tpos[..., 1].min()
    pad = np.zeros((J, D - 3))
    pad[:, :6] = (1, 0, 0, 0, 1, 0)
    tpos12 = np.concatenate([tpos, pad], axis=-1)
    mean = np.zeros((J, D))
    std = np.zeros((J, D))
    mean[0], std[0] = st['mean_root'], st['std_root']
    mean[1:], std[1:] = st['mean_local'], st['std_local']
    tn = np.nan_to_num((tpos12 - mean[None, :]) / std[None, :])[0] if False else np.nan_to_num((tpos12 - mean) / std)
    tpar = tn.copy()
    for j, p in enumerate(parents):
        if p != -1:
            tpar[j] = tn[p]
    rel, dist = compute_edge_relations_and_distances(parents)
    depths = compute_joint_depths(parents)
    spec = compute_laplacian_eigenvectors(parents, 8)

    def padj(x, tail, fill=0.0, dtype=np.float32):
        out = np.full((MAXJ,) + tail, fill, dtype=dtype)
        out[:x.shape[0]] = x
        return out
    P = {
        'tpos_first_frame': padj(tn, (D,)), 'tpos_first_frame_parents': padj(tpar, (D,)),
        'joint_depths': padj(depths, (), 0, np.int64), 'spectral_feats': padj(spec, (8,)),
        'joint_names_emb': padj(names_emb, (names_emb.shape[1],)),
        'mean': padj(mean, (D,)), 'std': padj(std, (D,), 1.0),
    }
    gd = np.zeros((MAXJ, MAXJ), np.int64)
    gd[:J, :J] = dist
    jr = np.zeros((MAXJ, MAXJ), np.int64)
    jr[:J, :J] = rel
    cond = {k: torch.from_numpy(v)[None] for k, v in P.items()}
    cond['graph_dist'] = torch.from_numpy(gd)[None]
    cond['joint_relations'] = torch.from_numpy(jr)[None]
    cond['n_joints'] = torch.tensor([J])
    cond['motion_length'] = torch.tensor([T])
    cond['caption_emb'] = torch.from_numpy(caption_emb.astype(np.float32))[None]
    for k, nm in (('tpos_first_frame', 'in_tpos'), ('tpos_first_frame_parents', 'in_tpos_parents'), ('joint_depths', 'in_depths'),
                  ('spectral_feats', 'in_spectral'), ('graph_dist', 'in_graph_dist'), ('joint_relations', 'in_relations'),
                  ('mean', 'in_mean'), ('std', 'in_std')):
        save(nm, cond[k][0])

    # ---- the model, its EMA weights
    model = UniMateGraphAdaLN(feature_len=D, max_motion_length=T, max_joints=MAXJ, max_depth=dc["max_depth"], latent_dim=mc["latent_dim"],
                              ff_size=mc["ff_size"], num_layers=mc["num_layers"], num_heads=mc["num_heads"], max_freqs=mc["max_freqs"],
                              text_dim=768, num_tpos_queries=mc["num_tpos_queries"]).eval()
    zf, root, ck = safeio.load_pth(os.path.join(a.unimate, a.ckpt))
    shadow = [safeio.tensor(zf, root, r) for r in ck["ema_state_dict"]["shadow_params"]]
    params = list(model.named_parameters())
    assert len(shadow) == len(params), (len(shadow), len(params))
    msd = ck["model_state_dict"]
    worst = 1.0
    for (n, p), s in zip(params, shadow):
        assert tuple(p.shape) == tuple(s.shape), (n, p.shape, s.shape)
        p.copy_(torch.from_numpy(s))
        # the order is the parameters()' only if each EMA tensor sits close to the model weight of the same name
        # (rope_j is shared: its weights are under the blocks' names in model_state_dict)
        mn = n if n in msd else "transformer_blocks.0.s_attn." + n.replace("rope_j", "rope")
        w = safeio.tensor(zf, root, msd[mn]).ravel().astype(np.float64)
        sv = s.ravel().astype(np.float64)
        if w.size > 4 and np.linalg.norm(w) > 0 and np.linalg.norm(sv) > 0:
            worst = min(worst, float(w @ sv / (np.linalg.norm(w) * np.linalg.norm(sv))))
    print(f"{len(params)} parameters, EMA vs model weights: lowest cosine {worst:.4f}")
    assert worst > 0.9, "the EMA tensors do not pair with the parameters in order"
    with open(os.path.join(a.out, "param_order.txt"), "w") as f:
        for n, p in params:
            f.write(f"{n} {' '.join(str(d) for d in p.shape)}\n")

    # ---- one pass
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn((1, MAXJ, D, T), generator=g)
    save("noise", noise[0])
    model.trace = {}
    tt = torch.tensor([0.37])
    oc = model(noise, tt, cond)
    for k, v in model.trace.items():
        save("pass_" + k, v[0])
    ou = model(noise, tt, cond, force_mask=True)
    model.trace = None
    guided = ou + a.cfg * (oc - ou)
    save("pass_cond", oc[0])
    save("pass_uncond", ou[0])
    save("pass_guided", guided[0])

    # ---- the flow
    nfe = [0]
    times = []

    def drift(t, x):
        nfe[0] += 1
        times.append(float(t))
        bt = torch.ones(x.size(0)) * t
        out = model(x, bt, cond)
        outu = model(x, bt, cond, force_mask=True)
        return outu + a.cfg * (out - outu)

    if not a.no_ode:
        from torchdiffeq import odeint
        ts = torch.linspace(0, 1, 50)
        sol = odeint(drift, noise, ts, method="dopri5", atol=[1e-6], rtol=[1e-3])[-1]
        save("ode", sol[0])
        x = sol[0].numpy().astype(np.float32)
    else:
        x = noise[0].numpy()
    # ---- decode (visualization.py: motion * std + mean; motion_utils.recover_unimate_anim_from_rot)
    mot = (x[:J].transpose(2, 0, 1) * P['std'][None, :J] + P['mean'][None, :J])          # [T, J, 12]
    mot = mot.astype(np.float64)
    R, rpos = recover_root(mot[:, 0])
    quats = hml_rotations_to_bvh_quaternions(mot[:, :, 3:9], parents)
    save("motion", mot.astype(np.float32))
    save("quats", quats.astype(np.float32))
    save("root_pos", rpos.astype(np.float32))
    json.dump({"joints": J, "prompt": a.prompt, "seed": a.seed, "cfg": a.cfg, "key": key, "nfe": nfe[0],
               "times": times, "parents": parents.tolist(), "names": names}, open(os.path.join(a.out, "meta.json"), "w"))
    print("done:", a.out, "ode evaluations", nfe[0])


if __name__ == "__main__":
    main()
