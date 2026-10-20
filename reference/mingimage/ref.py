"""Ming-Image-0.1-Design on the CPU in float32 - the reference for nextsycl's engine (image/mingimage): the stages'
tensors dumped for `nextsycl image check`. The weights are nextsycl's own files (the GGUFs dequantized, the VAE's
bf16), so both sides run the same numbers; the arithmetic is inclusionAI's (github.com/inclusionAI/Ming-Image,
3515cd7b): its DiT class (diffusion/transformer.py) runs as it is, the text side follows modeling_bailingmm2.py /
modeling_bailing_moe_v2.py line for line (the thinker in plain torch, its layers dequantized one at a time; the
connector is transformers' Qwen2 as the pipeline loads it), the VAE is Wan 2.1's decoder (ComfyUI's copy of Wan's
code, which names its tensors as the file does).

    CUDA_VISIBLE_DEVICES= python -I ref.py --te TE.gguf --dit DIT.gguf --vae VAE.safetensors --code MING_IMAGE_DIR \\
        --comfy COMFYUI_DIR --out DIR [--prompt TEXT] [--size 512] [--steps 12] [--seed 7] [--stages te,dit,vae]

Dumps (DIR, .npy float32 unless said): ids.npy (int32: the template's tokens, then <image>, 256 <imagePatch>,
</image>); prompt.txt; te_final [T, 2048] (the thinker's last hidden state, normed); cap [256, 2560] (the query
tokens through proj_in, the connector, proj_out); direct [L, 3840] (layers 5, 12 and the last over the L template
tokens, proj_directvlm); sigmas [steps + 1]; lat_0 [16, h, w] (the starting noise), and for steps 0 and 5: lat_i
(the input), vel_i (the DiT's output, before the pipeline's minus), at step 0 also nr_out [N, 3840] (the noise
refiner's last), cr_out [C, 3840] (the context refiner's last), l0_out [N + C, 3840] (layer 0's, image first);
lat_final [16, h, w]; rgba [4, H, W] (the VAE's output in [-1, 1]) and ref.png.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader
from gguf.quants import dequantize

torch.set_grad_enabled(False)

SYSTEM = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
IMAGE_START, IMAGE_PATCH, IMAGE_END = 157158, 157157, 157159
QUERY = 256


class G:
    def __init__(self, path):
        self.r = GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}

    def get(self, name):
        t = self.t[name]
        a = dequantize(t.data, t.tensor_type)
        shape = [int(x) for x in reversed(list(t.shape))]
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32).reshape(shape))


def save(out, name, t):
    a = t.detach().float().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t)
    np.save(os.path.join(out, name + ".npy"), a)


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


# -- the text side ----------------------------------------------------------------------------------------------

def tokens(te, prompt):
    from tokenizers import Tokenizer
    raw = bytes(np.asarray(te.t["tokenizer_json"].data).astype(np.uint8))
    tok = Tokenizer.from_str(raw.decode("utf-8"))
    j = json.loads(raw)
    print("tokenizer post-processor:", json.dumps(j.get("post_processor"))[:200])
    # processing_bailingmm2.apply_chat_template: system, eos, the user's turn, eos, the assistant's prefix
    text = SYSTEM + "<|role_end|>" + "<role>HUMAN</role>" + prompt + "<|role_end|>" + "<role>ASSISTANT</role>"
    ids = tok.encode(text, add_special_tokens=False).ids
    with_special = tok.encode(text).ids
    if with_special != ids:
        print("note: the tokenizer adds special tokens of its own:", with_special[:3], "...")
    # append_input_ids_with_multiscale_learnable_tokens (one 16 x 16 scale)
    return ids, ids + [IMAGE_START] + [IMAGE_PATCH] * QUERY + [IMAGE_END]


def positions(L, T):
    """get_t_scale_rope_index for one generated image after the L template tokens (the <image> token at L): text
    0..L, the block's 256 at t = h = L + 1 and w = L + 1 + (0..255) - 127, the end token after it."""
    t = torch.arange(T, dtype=torch.float64)
    h, w = t.clone(), t.clone()
    s = L + 1
    t[s:s + QUERY] = s
    h[s:s + QUERY] = s
    w[s:s + QUERY] = s + torch.arange(QUERY, dtype=torch.float64) - (QUERY - 1) // 2
    t[s + QUERY:] = h[s + QUERY:] = w[s + QUERY:] = s + 1 + torch.arange(T - s - QUERY, dtype=torch.float64)
    return t, h, w


def rope_tables(t, h, w, dim=64, theta=600000.0):
    """video_rope with mrope_section [8, 12, 12]: of the 32 frequencies, 0..23 alternate height / width, 24..31 time"""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    f = torch.empty(len(t), dim // 2, dtype=torch.float64)
    for k in range(dim // 2):
        p = (h if k % 2 == 0 else w) if k < 24 else t
        f[:, k] = p * inv[k]
    emb = torch.cat((f, f), dim=-1)
    return emb.cos().float(), emb.sin().float()


def gate_route(h, wg, bias, top_k=8, n_group=8, topk_group=4, scale=2.5):
    logits = h.float() @ wg.float().T
    scores = torch.sigmoid(logits)
    routing = scores + bias
    n = routing.shape[0]
    group_scores = routing.view(n, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(n, n_group, routing.shape[1] // n_group).reshape(n, -1)
    masked = routing.masked_fill(~score_mask.bool(), float("-inf"))
    idx = torch.topk(masked, k=top_k, dim=-1, sorted=False)[1]
    wts = torch.gather(scores, 1, idx)
    wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * scale
    return idx, wts


def thinker(te, ids_all, L, out=None):
    T = len(ids_all)
    emb = te.get("thinker.embed_tokens.weight")
    x = emb[torch.tensor(ids_all).clamp(0, emb.shape[0] - 1)].clone()
    del emb
    x[L + 1:L + 1 + QUERY] = te.get("query_tokens")
    image_mask = torch.zeros(T, dtype=torch.bool)
    image_mask[L + 1:L + 1 + QUERY] = True
    cos, sin = rope_tables(*positions(L, T))
    causal = torch.full((T, T), float("-inf")).triu(1)
    captured = []
    for i in range(20):
        t0 = time.time()
        if i in (5, 12):
            captured.append(x.clone())
        p = f"thinker.layers.{i}."
        g = lambda n: te.get(p + n)
        hh = rmsnorm(x, g("input_layernorm.weight"), 1e-6)
        qkv = (hh @ g("attention.query_key_value.weight").T).view(T, 24, 128)
        q, k, v = qkv[:, :16], qkv[:, 16:20], qkv[:, 20:24]
        q = rmsnorm(q, g("attention.q_norm.weight"), 1e-6)
        k = rmsnorm(k, g("attention.k_norm.weight"), 1e-6)
        c, s = cos[:, None, :], sin[:, None, :]
        q = torch.cat((q[..., :64] * c + rotate_half(q[..., :64]) * s, q[..., 64:]), dim=-1)
        k = torch.cat((k[..., :64] * c + rotate_half(k[..., :64]) * s, k[..., 64:]), dim=-1)
        k = k.repeat_interleave(4, dim=1)
        v = v.repeat_interleave(4, dim=1)
        sc = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(128) + causal
        a = torch.einsum("hqk,khd->qhd", torch.softmax(sc, dim=-1), v).reshape(T, 2048)
        x = x + a @ g("attention.dense.weight").T
        hh = rmsnorm(x, g("post_attention_layernorm.weight"), 1e-6)
        if i == 0:
            m = (F.silu(hh @ g("mlp.gate_proj.weight").T) * (hh @ g("mlp.up_proj.weight").T)) @ g("mlp.down_proj.weight").T
        else:
            ti, tw = gate_route(hh, g("mlp.gate.proj.weight"), g("mlp.gate.expert_bias"))
            ii, iw = gate_route(hh, g("mlp.image_gate.proj.weight"), g("mlp.image_gate.expert_bias"))
            idx = torch.where(image_mask[:, None], ii, ti)
            if out:
                np.save(os.path.join(out, f"te_route{i}.npy"), torch.sort(idx, dim=-1)[0].numpy().astype(np.int32))
            wts = torch.where(image_mask[:, None], iw, tw)
            gu = g("mlp.experts.gate_up_proj.weight")  # [256, 1024, 2048]
            dn = g("mlp.experts.down_proj.weight")  # [256, 2048, 512]
            m = torch.zeros_like(hh)
            for e in torch.unique(idx).tolist():
                rows, slot = (idx == e).nonzero(as_tuple=True)
                y = hh[rows] @ gu[e].T
                y = F.silu(y[:, :512]) * y[:, 512:]
                m.index_add_(0, rows, (y @ dn[e].T) * wts[rows, slot][:, None])
            sh = (F.silu(hh @ g("mlp.shared_experts.gate_proj.weight").T) * (hh @ g("mlp.shared_experts.up_proj.weight").T))
            m = m + sh @ g("mlp.shared_experts.down_proj.weight").T
        x = x + m
        if out:
            save(out, f"te_x{i}", x)
        print(f"thinker layer {i}: {time.time() - t0:.1f} s", flush=True)
    final = rmsnorm(x, te.get("thinker.norm.weight"), 1e-6)
    captured.append(final)
    return final, captured


def connector(te, q):
    """proj_in, transformers' Qwen2 (the pipeline's AutoModelForCausalLM, an all-ones 4-D mask: no causality), its
    last hidden state (normed), proj_out"""
    from transformers import Qwen2Config, Qwen2Model
    cfg = Qwen2Config(hidden_size=1536, intermediate_size=8960, num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2,
                      rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=32768, vocab_size=151936, hidden_act="silu",
                      tie_word_embeddings=True, attn_implementation="eager")
    m = Qwen2Model(cfg).float().eval()
    sd = {k[len("connector."):]: te.get(k) for k in te.t if k.startswith("connector.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert unexpected == [] and set(missing) <= {"embed_tokens.weight"}, (missing, unexpected)
    x = q @ te.get("proj_in.weight").T + te.get("proj_in.bias")
    out = m(inputs_embeds=x[None], attention_mask=torch.ones(1, 1, x.shape[0], x.shape[0]), output_hidden_states=True).hidden_states[-1][0]
    # the same by hand, bidirectional: the mask's meaning checked
    pos = torch.arange(x.shape[0], dtype=torch.float64)
    inv = 1.0 / (1e6 ** (torch.arange(0, 128, 2, dtype=torch.float64) / 128))
    f = pos[:, None] * inv[None]
    cos, sin = torch.cat((f, f), -1).cos().float()[:, None], torch.cat((f, f), -1).sin().float()[:, None]
    hcur = x
    for i in range(28):
        p = f"layers.{i}."
        hh = rmsnorm(hcur, sd[p + "input_layernorm.weight"], 1e-6)
        q_ = (hh @ sd[p + "self_attn.q_proj.weight"].T + sd[p + "self_attn.q_proj.bias"]).view(-1, 12, 128)
        k_ = (hh @ sd[p + "self_attn.k_proj.weight"].T + sd[p + "self_attn.k_proj.bias"]).view(-1, 2, 128)
        v_ = (hh @ sd[p + "self_attn.v_proj.weight"].T + sd[p + "self_attn.v_proj.bias"]).view(-1, 2, 128)
        q_ = q_ * cos + rotate_half(q_) * sin
        k_ = (k_ * cos + rotate_half(k_) * sin).repeat_interleave(6, 1)
        v_ = v_.repeat_interleave(6, 1)
        a = torch.einsum("hqk,khd->qhd", torch.softmax(torch.einsum("qhd,khd->hqk", q_, k_) / math.sqrt(128), -1), v_).reshape(-1, 1536)
        hcur = hcur + a @ sd[p + "self_attn.o_proj.weight"].T
        hh = rmsnorm(hcur, sd[p + "post_attention_layernorm.weight"], 1e-6)
        hcur = hcur + (F.silu(hh @ sd[p + "mlp.gate_proj.weight"].T) * (hh @ sd[p + "mlp.up_proj.weight"].T)) @ sd[p + "mlp.down_proj.weight"].T
    mine = rmsnorm(hcur, sd["norm.weight"], 1e-6)
    print(f"connector: transformers vs bidirectional by hand: max |d| {float((out - mine).abs().max()):.3e} (|x| {float(out.abs().max()):.2f})")
    return out @ te.get("proj_out.weight").T + te.get("proj_out.bias")


# -- the DiT ------------------------------------------------------------------------------------------------------

def load_dit(path, code):
    sys.path.insert(0, code)
    from diffusion.transformer import DiffusionTransformer
    g = G(path)
    with torch.device("meta"):
        m = DiffusionTransformer(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=3840, n_layers=30, n_refiner_layers=2,
                                 n_heads=30, n_kv_heads=30, norm_eps=1e-5, qk_norm=True, cap_feat_dim=2560, rope_theta=256.0,
                                 t_scale=1000.0, axes_dims=[32, 48, 48], axes_lens=[20480, 512, 512],
                                 alignment_padding_mode="zero_masked", multi_frame_output=False)
    big = lambda n: n.startswith("layers.") or n.startswith("noise_refiner.") or n.startswith("context_refiner.")
    # the blocks kept in half (the Q8_0 values as the GPU makes them), float32 while each runs; the rest float32
    sd = {n: (g.get(n).half() if big(n) else g.get(n)) for n in g.t}
    m.load_state_dict(sd, strict=True, assign=True)
    m.eval()
    for b in list(m.layers) + list(m.noise_refiner) + list(m.context_refiner):
        b.register_forward_pre_hook(lambda mod, args: mod.float() and None)
        b.register_forward_hook(lambda mod, args, out: mod.half() and None)
    return m


# -- the VAE ------------------------------------------------------------------------------------------------------

def decode(vae_path, comfy, lat):
    sys.path.insert(0, comfy)
    from safetensors.torch import load_file
    # ComfyUI opens CUDA at import unless told --cpu (its own argument parser reads sys.argv)
    import comfy.options
    comfy.options.enable_args_parsing(True)
    saved, sys.argv = sys.argv, ["ref.py", "--cpu"]
    from comfy.ldm.wan.vae import WanVAE
    sys.argv = saved
    sd = {k: v.float() for k, v in load_file(vae_path).items()}
    v = WanVAE(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True],
               image_channels=4, conv_out_channels=4).float().eval()
    missing, unexpected = v.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k.startswith("decoder") or k.startswith("conv2")], missing
    z = (lat / 8.0064)[None, :, None]  # the pipeline: latents / scaling_factor + shift_factor (0)
    x = v.conv2(z)
    out = v.decoder(x[:, :, :1], feat_cache=None, feat_idx=[0])[0]
    return out[0, :, 0].clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--code", required=True, help="the Ming-Image repository (its diffusion/transformer.py)")
    ap.add_argument("--comfy", required=True, help="a ComfyUI checkout (Wan's VAE code)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="A flat vector poster for a lemonade stand: a yellow lemon, the words \"FRESH LEMONADE\" in bold blue letters, white background")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stages", default="te,dit,vae")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stages = a.stages.split(",")
    torch.manual_seed(a.seed)
    open(os.path.join(a.out, "prompt.txt"), "w").write(a.prompt)

    if "te" in stages:
        te = G(a.te)
        ids, ids_all = tokens(te, a.prompt)
        L = len(ids)
        np.save(os.path.join(a.out, "ids.npy"), np.array(ids_all, dtype=np.int32))
        print(f"{L} template tokens, {len(ids_all)} in all")
        final, captured = thinker(te, ids_all, L, a.out)
        save(a.out, "te_final", final)
        cap = connector(te, final[L + 1:L + 1 + QUERY])
        save(a.out, "cap", cap)
        direct = torch.cat([c[:L] for c in captured], dim=-1)
        direct = rmsnorm(direct, te.get("proj_directvlm.0.weight"), 1e-5) @ te.get("proj_directvlm.1.weight").T + te.get("proj_directvlm.1.bias")
        save(a.out, "direct", direct)
        del te
    cap = torch.from_numpy(np.load(os.path.join(a.out, "cap.npy")))
    direct = torch.from_numpy(np.load(os.path.join(a.out, "direct.npy")))

    if "dit" in stages:
        m = load_dit(a.dit, a.code)
        h = w = a.size // 8
        # FlowMatchEulerDiscreteScheduler (shift 6, the pipeline's sigma_min 0; its dynamic-shift switch never reaches
        # the scheduler: the config is a FrozenDict) - linspace(1, 0, steps) shifted, then 0
        lin = np.linspace(1.0, 0.0, a.steps)
        sig = np.concatenate([6.0 * lin / (1 + 5.0 * lin), [0.0]]).astype(np.float32)
        save(a.out, "sigmas", torch.from_numpy(sig))
        lat = torch.randn(16, h, w, generator=torch.Generator().manual_seed(a.seed))
        save(a.out, "lat_0", lat)
        grabs = {}
        # (a forward hook that returns a value replaces the module's output: these return None)
        def grab(key):
            def hook(mod, ar, out):
                grabs.setdefault(key, out[0].clone())
            return hook
        hooks = [m.noise_refiner[-1].register_forward_hook(grab("nr_out")),
                 m.context_refiner[-1].register_forward_hook(grab("cr_out")),
                 m.layers[0].register_forward_hook(grab("l0_out"))]
        for i in range(a.steps):
            t0 = time.time()
            s, sn = float(sig[i]), float(sig[i + 1])
            out = m([lat[:, None].float()], torch.tensor([1.0 - s]), [cap.float()], cap_feats_2=[direct.float()], return_dict=False)[0][0][:, 0]
            if i in (0, 5):
                save(a.out, f"lat_{i}", lat)
                save(a.out, f"vel_{i}", out)
            if i == 0:
                for k, v in grabs.items():
                    save(a.out, k, v)
                for hk in hooks:
                    hk.remove()
            lat = lat + (sn - s) * (-out)
            print(f"step {i}: sigma {s:.4f} in {time.time() - t0:.1f} s", flush=True)
        save(a.out, "lat_final", lat)
        del m

    if "vae" in stages:
        lat = torch.from_numpy(np.load(os.path.join(a.out, "lat_final.npy")))
        rgba = decode(a.vae, a.comfy, lat)
        save(a.out, "rgba", rgba)
        from PIL import Image
        px = ((rgba.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(px, "RGBA").save(os.path.join(a.out, "ref.png"))
        print("alpha min / mean:", int(px[..., 3].min()), float(px[..., 3].mean()))


if __name__ == "__main__":
    main()
