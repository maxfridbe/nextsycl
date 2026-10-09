"""reference/qwenimage21/ref.py: Qwen-Image 2.1 on the CPU, from the SAME quantized files nextsycl serves, dumping
every stage the engine is checked against.

    python ref.py te   --te <qwen3vl int8_convrot.safetensors> --cfg <Qwen-Image-2.1 dir> --prompt TEXT --out DIR
    python ref.py dit  --dit <transformer .gguf> --cfg DIR --out DIR --size 512 --steps 4 --seed 7
    python ref.py vae  --vae <vae .safetensors> --cfg DIR --out DIR

The stages run one at a time (each frees its weights) and talk through DIR:
  te:  tokens.json (ids, the image-pad mask), embeds.npy [S, 4096] - the last layer's hidden state before the final
       norm, the system prompt's tokens dropped (the pipeline's _get_qwen_prompt_embeds)
  dit: noise.npy [N, 64] (the starting latents, float32, from --seed), sigmas.npy, block0.npy (block 0's output at
       step 0), v0.npy (step 0's velocity), latents_<i>.npy after each step
  vae: image.npy [H, W, 4] in [-1, 1] and image.png

Weights: the transformer GGUF dequantized exactly (Q8_0 / BF16 -> float32); the text encoder through comfy-kitchen's
own int8 ConvRot ops (what ComfyUI runs on this file: rotated activations quantized per row, int32 products, row
scales); the VAE's BF16 weights as float32. Activations are float32 throughout, so a difference from the engine is
the engine's, not the reference's rounding.
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def st_tensor(path, h, base, name):
    e = h[name]
    a, b = e["data_offsets"]
    dt = {"F32": np.float32, "I8": np.int8, "U8": np.uint8, "BF16": np.uint16, "F16": np.float16}[e["dtype"]]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = np.frombuffer(f.read(b - a), dtype=dt).reshape(e["shape"])
    if e["dtype"] == "BF16":
        return torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32).copy())
    return torch.from_numpy(raw.copy())


# ------------------------------------------------------------------------------------------------ text encoder
def run_te(a):
    from comfy_kitchen.backends.eager import quantization as ck
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(a.cfg, "processor"))
    cfg = json.load(open(os.path.join(a.cfg, "text_encoder", "config.json")))
    tc = cfg.get("text_config", cfg)
    L, D, H, KV, HD = tc["num_hidden_layers"], tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    eps, theta = tc["rms_norm_eps"], tc["rope_theta"]
    sysp = "Comprehend and analyze the provided prompt."
    text = f"<|im_start|>system\n{sysp}<|im_end|>\n<|im_start|>user\n{a.prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text)["input_ids"]
    drop = len(tok(f"<|im_start|>system\n{sysp}<|im_end|>\n")["input_ids"])
    h, base = st_header(a.te)
    W = lambda n: st_tensor(a.te, h, base, n)

    def lin(x, name):
        return ck.int8_linear(x, W(name + ".weight"), W(name + ".weight_scale"), out_dtype=torch.float32, convrot=True, convrot_groupsize=256)

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    idx = torch.tensor(ids)
    x = ck.dequantize_int8_embedding(W("model.embed_tokens.weight"), W("model.embed_tokens.weight_scale"), idx, 256, 0).float() \
        if hasattr(ck, "DTYPE_CODE_TO_DTYPE") and ck.DTYPE_CODE_TO_DTYPE.get(0) == torch.float32 else None
    if x is None:
        q = torch.nn.functional.embedding(idx, W("model.embed_tokens.weight")).float() * torch.nn.functional.embedding(idx, W("model.embed_tokens.weight_scale"))
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight
        x = _rotate_weight(q, _build_hadamard(256), 256)
    S = x.shape[0]
    pos = torch.arange(S, dtype=torch.float64)
    inv = 1.0 / (theta ** (torch.arange(0, HD, 2, dtype=torch.float64) / HD))
    ang = torch.outer(pos, inv)  # text-only positions: the three M-RoPE axes coincide, so this is plain RoPE
    cos, sin = torch.cos(ang).float(), torch.sin(ang).float()

    def rope(t):  # t [S, heads, HD], rotate-half form (HF Qwen3)
        t1, t2 = t[..., : HD // 2], t[..., HD // 2:]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], dim=-1)

    mask = torch.full((S, S), float("-inf")).triu(1)
    for l in range(L):
        p = f"model.layers.{l}."
        hh = rms(x, W(p + "input_layernorm.weight"))
        q = lin(hh, p + "self_attn.q_proj").view(S, H, HD)
        k = lin(hh, p + "self_attn.k_proj").view(S, KV, HD)
        v = lin(hh, p + "self_attn.v_proj").view(S, KV, HD)
        q = rope(rms(q, W(p + "self_attn.q_norm.weight")))
        k = rope(rms(k, W(p + "self_attn.k_norm.weight")))
        k = k.repeat_interleave(H // KV, dim=1)
        v = v.repeat_interleave(H // KV, dim=1)
        att = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(HD) + mask
        o = torch.einsum("hqk,khd->qhd", att.softmax(-1), v).reshape(S, H * HD)
        x = x + lin(o, p + "self_attn.o_proj")
        hh = rms(x, W(p + "post_attention_layernorm.weight"))
        x = x + lin(torch.nn.functional.silu(lin(hh, p + "mlp.gate_proj")) * lin(hh, p + "mlp.up_proj"), p + "mlp.down_proj")
        if l == 0:
            np.save(os.path.join(a.out, "te_layer0.npy"), x.numpy())
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    json.dump({"ids": ids, "drop": drop, "image_pad": [int(i == pad) for i in ids[drop:]]}, open(os.path.join(a.out, "tokens.json"), "w"))
    np.save(os.path.join(a.out, "embeds.npy"), x[drop:].numpy())
    print(f"te: {S} tokens ({drop} dropped), embeds {tuple(x[drop:].shape)}")


# ------------------------------------------------------------------------------------------------ transformer
def load_gguf_into(model, path):
    """Each GGUF tensor dequantized straight into the model's float32 parameter: the peak is the model plus one tensor
    (a dict of them all first was 47 GB on a 61 GB box)."""
    from gguf import GGUFReader
    from gguf.quants import dequantize

    params = dict(model.named_parameters())
    seen = set()
    for t in GGUFReader(path).tensors:
        if t.name not in params:
            sys.exit(f"dit: {t.name} is not a parameter of the model")
        arr = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)
        p = params[t.name]
        p.data.copy_(torch.from_numpy(arr.reshape(p.shape)))
        seen.add(t.name)
        del arr
    missing = [n for n in params if n not in seen]
    if missing:
        sys.exit(f"dit: parameters the file does not have: {missing[:5]}")


def run_dit(a):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps

    cfg = QwenImage21Transformer2DModel.load_config(os.path.join(a.cfg, "transformer"))
    with torch.device("meta"):
        model = QwenImage21Transformer2DModel.from_config(cfg)
    model = model.to_empty(device="cpu").float()
    # the buffers the constructor computes (RoPE tables, timestep frequencies) are not in the file: rebuild them
    fresh = QwenImage21Transformer2DModel.from_config({**cfg, "num_layers": 1})
    model.pos_embed = fresh.pos_embed
    model.time_text_embed.time_proj = fresh.time_text_embed.time_proj
    del fresh
    load_gguf_into(model, a.dit)
    model = model.eval()
    emb = torch.from_numpy(np.load(os.path.join(a.out, "embeds.npy")))[None]
    toks = json.load(open(os.path.join(a.out, "tokens.json")))
    hw = a.size // 16
    n = hw * hw
    g = torch.Generator().manual_seed(a.seed)
    lat = torch.randn((1, 64, hw, hw), generator=g, dtype=torch.float32).view(1, 64, n).transpose(1, 2).contiguous()
    np.save(os.path.join(a.out, "noise.npy"), lat[0].numpy())
    sched = FlowMatchEulerDiscreteScheduler.from_config(FlowMatchEulerDiscreteScheduler.load_config(os.path.join(a.cfg, "scheduler")))
    sig = np.linspace(1.0, 1 / a.steps, a.steps)
    mu = calculate_shift(n, sched.config.get("base_image_seq_len", 256), sched.config.get("max_image_seq_len", 4096),
                         sched.config.get("base_shift", 0.5), sched.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(sched, a.steps, "cpu", sigmas=sig, mu=mu)
    np.save(os.path.join(a.out, "sigmas.npy"), sched.sigmas.numpy())
    img_mask = torch.tensor(toks["image_pad"] + [1] * (n // 4), dtype=torch.bool)[None]
    shapes = [[(1, hw, hw)]]
    cache = None
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    cache = QwenImage21KVCache(len(model.transformer_blocks))
    out0 = {}
    hook = model.transformer_blocks[0].register_forward_hook(lambda m, i, o: out0.setdefault("b0", o.detach().clone()))
    sched.set_begin_index(0)
    for i, t in enumerate(timesteps):
        v = model(hidden_states=lat, timestep=(t.expand(1) / 1000).float(), encoder_hidden_states=emb, img_shapes=shapes,
                  img_mask=img_mask, kv_cache=cache, kv_cache_mode="extract" if i == 0 else "cached", return_dict=False)[0]
        v = v[:, -n:]
        if i == 0:
            hook.remove()
            np.save(os.path.join(a.out, "block0.npy"), out0["b0"][0].numpy())
            np.save(os.path.join(a.out, "v0.npy"), v[0].numpy())
        lat = sched.step(v, t, lat, return_dict=False)[0]
        np.save(os.path.join(a.out, f"latents_{i}.npy"), lat[0].numpy())
        print(f"dit: step {i + 1}/{a.steps} (t {float(t):.1f})", flush=True)
    json.dump({"size": a.size, "steps": a.steps, "seed": a.seed, "mu": mu}, open(os.path.join(a.out, "dit.json"), "w"))


# ------------------------------------------------------------------------------------------------ VAE
def vae_name(n):
    """The original (Wan / ComfyUI) name of a decoder tensor -> diffusers' (None: the encoder's, not needed)"""
    import re
    if n.startswith("encoder") or n.startswith("conv1"):
        return None
    if n.startswith("conv2."):
        return "post_quant_conv." + n[len("conv2."):]
    n = n.replace("decoder.conv1.", "decoder.conv_in.")
    n = n.replace("decoder.head.0.", "decoder.norm_out.").replace("decoder.head.2.", "decoder.conv_out.")
    res = {"0": "norm1.gamma", "2": "conv1", "3": "norm2.gamma", "6": "conv2"}

    def residual(prefix, m):
        part = res[m.group(1)]
        rest = m.group(2)
        return prefix + (part if part.endswith("gamma") else part + rest)

    m = re.match(r"decoder\.middle\.([02])\.residual\.(\d)(.*)", n)
    if m:
        i = "0" if m.group(1) == "0" else "1"
        part = res[m.group(2)]
        return f"decoder.mid_block.resnets.{i}." + (part if part.endswith("gamma") else part + m.group(3))
    m = re.match(r"decoder\.middle\.1\.(.*)", n)
    if m:
        return "decoder.mid_block.attentions.0." + m.group(1)
    m = re.match(r"decoder\.upsamples\.(\d)\.upsamples\.(\d)\.residual\.(\d)(.*)", n)
    if m:
        part = res[m.group(3)]
        return f"decoder.up_blocks.{m.group(1)}.resnets.{m.group(2)}." + (part if part.endswith("gamma") else part + m.group(4))
    m = re.match(r"decoder\.upsamples\.(\d)\.upsamples\.(\d)\.shortcut(.*)", n)
    if m:
        return f"decoder.up_blocks.{m.group(1)}.resnets.{m.group(2)}.conv_shortcut{m.group(3)}"
    m = re.match(r"decoder\.upsamples\.(\d)\.upsamples\.3\.(resample|time_conv)(.*)", n)
    if m:
        return f"decoder.up_blocks.{m.group(1)}.upsampler.{m.group(2)}{m.group(3)}"
    return n


def run_vae(a):
    from diffusers import AutoencoderKLQwenImage21

    vcfg = AutoencoderKLQwenImage21.load_config(os.path.join(a.cfg, "vae"))
    with torch.device("meta"):
        vae = AutoencoderKLQwenImage21.from_config(vcfg)
    vae = vae.to_empty(device="cpu").float().eval()
    params = dict(vae.named_parameters())
    h, base = st_header(a.vae)
    done = set()
    for name in h:
        tgt = vae_name(name)
        if tgt is None:
            continue
        if tgt not in params:
            sys.exit(f"vae: {name} -> {tgt} is not a parameter")
        t = st_tensor(a.vae, h, base, name).float()
        p = params[tgt]
        p.data.copy_(t.reshape(p.shape))
        done.add(tgt)
    missing = [n for n in params if n not in done and (n.startswith("decoder") or n.startswith("post_quant"))]
    if missing:
        sys.exit(f"vae: decoder parameters the file does not have: {missing[:5]}")
    meta = json.load(open(os.path.join(a.out, "dit.json")))
    hw = meta["size"] // 16
    lat = torch.from_numpy(np.load(os.path.join(a.out, f"latents_{meta['steps'] - 1}.npy")))
    lat = lat.transpose(0, 1).reshape(1, 64, 1, hw, hw)
    mean = torch.tensor(vcfg["latents_mean"]).view(1, 64, 1, 1, 1)
    std = torch.tensor(vcfg["latents_std"]).view(1, 64, 1, 1, 1)
    img = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]  # [1, 4, H, W] in [-1, 1]
    arr = img[0].permute(1, 2, 0).numpy()
    np.save(os.path.join(a.out, "image.npy"), arr)
    from PIL import Image
    rgba = ((arr / 2 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(os.path.join(a.out, "image.png"))
    print(f"vae: image {arr.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["te", "dit", "vae"])
    ap.add_argument("--te"); ap.add_argument("--dit"); ap.add_argument("--vae"); ap.add_argument("--cfg", required=True)
    ap.add_argument("--prompt", default="A red fox sitting in fresh snow, morning light, photograph")
    ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"te": run_te, "dit": run_dit, "vae": run_vae}[a.stage](a)
