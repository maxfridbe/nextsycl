"""Z-Image-Turbo on the CPU in float32 - the reference for nextsycl's engine (image/zimage): the stages' tensors dumped
for `nextsycl image check`. diffusers' own classes at commit 84532a058702c74ae5d1dd6ceb05683fc5d5e768 (main,
pipelines/z_image/pipeline_z_image.py and models/transformers/transformer_z_image.py) run as they are, on the files the
engine loads: the DiT's GGUF (ComfyUI's names, renamed to diffusers' and its fused qkv split; each matrix kept as
stored and expanded to float32 for its own product only), the text encoder's single safetensors file (ComfyUI's
qwen_3_4b, into transformers' Qwen3Model) and the VAE's diffusers file. The denoising loop is the pipeline's, copied
so that every step can be dumped.

    python -I ref.py --dit DIT.gguf --te TE.safetensors --hf HF_DIR --out DIR [--prompt TEXT] [--size 1024x1024]
                     [--steps 8] [--seed 42] [--stages te,dit,vae] [--threads 16]
    python -I ref.py --files --dit DIT.gguf --te TE.safetensors --hf HF_DIR      (the files against the official ones)

HF_DIR: Tongyi-MAI/Z-Image-Turbo at f332072a (its configs, tokenizer and VAE; with --files its float32 transformer
and its text encoder's shards too).

Dumps (DIR, .npy float32 unless said): prompt.txt; text.txt (the chat template); ids.npy (int32: the template's
tokens, the pipeline's attention-masked rows); te_h1, te_h18, te_h35 (= cap, the pipeline's hidden_states[-2]),
te_h36 [L, 2560] (hidden_states[k]: k = 0 the embeddings, 36 the last layer's after the final norm); sigmas
[steps + 1]; lat_0 [16, h, w] (torch.randn from the seed); per step i: lat_i (the input), vel_i (the DiT's output,
before the pipeline's minus); at step 0 also t_emb [256], x_emb [N', 3840] (the patches embedded, the pad token in the
image's padding), nr_out [N', 3840] (the noise refiner's last), cap_emb [C', 3840] (embedded, the pad token in the
caption's padding), cr_out [C', 3840] (the context refiner's last), l0_out and l29_out [N' + C', 3840] (image first)
and stats.json (each block linear's largest |input| and |output|: where half's range is passed); N' and C' are N and
the caption's length rounded up to 32; lat_final [16, h, w]; rgb [3, H, W] (the VAE's output, in [-1, 1] as it comes)
and ref.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

MAX_LEN = 512


def save(out, name, t):
    if torch.is_tensor(t):
        t = t.detach().contiguous()
        a = (t if t.dtype == torch.int32 else t.float()).numpy()
    else:
        a = np.asarray(t)
    np.save(os.path.join(out, name + ".npy"), a)
    if a.dtype == np.float32:
        print(f"  {name} {list(a.shape)} mean {a.mean():.5f} std {a.std():.5f}", flush=True)


# ---- the DiT's GGUF: ComfyUI's names to diffusers'

def gguf_tensors(path):
    """name (diffusers') -> a function returning the float32 tensor; matrices stay as stored until asked"""
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        shape = [int(x) for x in reversed(list(t.shape))]
        ty = t.tensor_type

        def get(t=t, shape=shape, ty=ty):
            return torch.from_numpy(np.ascontiguousarray(dequantize(t.data, ty), dtype=np.float32).reshape(shape))

        name = t.name
        if name.startswith("x_embedder."):
            name = "all_x_embedder.2-1." + name[len("x_embedder."):]
        elif name.startswith("final_layer."):
            name = "all_final_layer.2-1." + name[len("final_layer."):]
        name = name.replace(".attention.out.", ".attention.to_out.0.").replace(".attention.q_norm.", ".attention.norm_q.") \
                   .replace(".attention.k_norm.", ".attention.norm_k.")
        if name.endswith(".attention.qkv.weight"):
            p = name[: -len("qkv.weight")]
            d = shape[0] // 3
            for i, part in enumerate(["to_q", "to_k", "to_v"]):
                out[f"{p}{part}.weight"] = (lambda g=get, i=i, d=d: g()[i * d:(i + 1) * d])
            continue
        if name in ("x_pad_token", "cap_pad_token"):
            out[name] = (lambda g=get: g().reshape(1, -1))
            continue
        out[name] = get
    return out


# each block linear's largest |input| and |output| while recording (step 0): where half's range (65,504) is passed
STATS = None


class LazyLinear(nn.Module):
    """A linear whose weight is made float32 for its own product only (the GGUF's matrices, 6B of them)"""

    def __init__(self, getw, bias, name):
        super().__init__()
        self.getw = getw
        self.name = name
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        y = F.linear(x.float(), self.getw(), self.bias)
        if STATS is not None:
            STATS[self.name] = [round(x.abs().max().item(), 2), round(y.abs().max().item(), 2)]
        return y


def load_dit(path, hf):
    from diffusers import ZImageTransformer2DModel
    from accelerate import init_empty_weights
    cfg = json.load(open(os.path.join(hf, "transformer", "config.json")))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with init_empty_weights():
        m = ZImageTransformer2DModel(**cfg)
    w = gguf_tensors(path)
    used = set()
    for mname, mod in list(m.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            # the blocks' matrices lazily; the embedders' and the final layer's (small, and read for their dtype) as they are
            if isinstance(child, nn.Linear) and full.startswith(("layers.", "noise_refiner.", "context_refiner.")):
                wk, bk = full + ".weight", full + ".bias"
                if wk not in w:
                    raise SystemExit(f"the GGUF has no {wk}")
                bias = w[bk]() if child.bias is not None else None
                setattr(mod, cname, LazyLinear(w[wk], bias, full))
                used.update([wk, bk])
    # every other parameter (norms, pad tokens) as float32
    for name, p in list(m.named_parameters()):
        if name in used:
            continue
        if name not in w:
            raise SystemExit(f"the GGUF has no {name}")
        t = w[name]()
        if list(t.shape) != list(p.shape):
            raise SystemExit(f"{name}: {list(t.shape)} in the GGUF, {list(p.shape)} in the model")
        mod_name, _, attr = name.rpartition(".")
        mod = m.get_submodule(mod_name) if mod_name else m
        setattr(mod, attr, nn.Parameter(t, requires_grad=False))
        used.add(name)
    left = [k for k in w if k not in used]
    if left:
        raise SystemExit(f"tensors the model did not take: {left[:8]}")
    return m.eval()


# ---- the text side

def template(hf, prompt):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(hf, "tokenizer"))
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ti = tok([text], padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt")
    return text, ti.input_ids, ti.attention_mask.bool()


def load_te(path, hf):
    """transformers' Qwen3Model on the file's tensors: the linears' bfloat16 matrices kept so and made float32 for their
    own product (as the DiT's), everything else float32"""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3Model
    cfg = Qwen3Config.from_pretrained(os.path.join(hf, "text_encoder"))
    with init_empty_weights():
        m = Qwen3Model(cfg)
    sd = {k[len("model."):]: v for k, v in load_file(path).items() if k.startswith("model.")}
    used = set()
    for mname, mod in list(m.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{mname}.{cname}" if mname else cname
                w = sd[full + ".weight"]
                setattr(mod, cname, LazyLinear(lambda w=w: w.float(), None, full))
                used.add(full + ".weight")
    for name, p in list(m.named_parameters()):
        if name in used:
            continue
        mod_name, _, attr = name.rpartition(".")
        setattr(m.get_submodule(mod_name) if mod_name else m, attr, nn.Parameter(sd[name].float(), requires_grad=False))
        used.add(name)
    left = [k for k in sd if k not in used]
    if left:
        raise SystemExit(f"text encoder: tensors the model did not take: {left[:4]}")
    return m.eval()


# ---- the official files against ours

def files(a):
    from safetensors import safe_open
    w = gguf_tensors(a.dit)
    shards = sorted(f for f in os.listdir(os.path.join(a.hf, "transformer")) if f.endswith(".safetensors"))
    exact, worst, n, bf16_ok = 0, (0.0, ""), 0, True
    for s in shards:
        with safe_open(os.path.join(a.hf, "transformer", s), "pt") as f:
            for k in f.keys():
                ref = f.get_tensor(k)
                if ref.dtype == torch.float32:
                    low = (ref.view(torch.int32) & 0xFFFF) != 0
                    if low.any():
                        bf16_ok = False
                ours = w[k]().reshape(ref.shape)
                d = (ours - ref.float()).abs().max().item()
                rel = ((ours - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()
                n += 1
                exact += d == 0
                if rel > worst[0]:
                    worst = (rel, k)
    print(f"DiT: {n} official tensors ({'every value a bfloat16' if bf16_ok else 'float32 values beyond bfloat16'}); "
          f"{exact} identical in the GGUF, the worst relative difference {worst[0]:.3e} ({worst[1]})")
    from safetensors.torch import load_file
    te = load_file(a.te)
    exact, n = 0, 0
    for s in sorted(f for f in os.listdir(os.path.join(a.hf, "text_encoder")) if f.endswith(".safetensors")):
        with safe_open(os.path.join(a.hf, "text_encoder", s), "pt") as f:
            for k in f.keys():
                n += 1
                ours = te.get(k)
                exact += ours is not None and ours.dtype == f.get_tensor(k).dtype and torch.equal(ours, f.get_tensor(k))
    print(f"text encoder: {exact} of {n} official tensors identical in {os.path.basename(a.te)} (it holds {len(te)})")


def main():
    global STATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", required=True)
    ap.add_argument("--te", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out")
    ap.add_argument("--prompt", default="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
                    "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
                    "lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted "
                    "tiered pagoda (西安大雁塔), blurred colorful distant lights.")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stages", default="te,dit,vae")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--files", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.files:
        files(a)
        return
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, get_default_z_image_sigmas, retrieve_timesteps
    out = a.out
    os.makedirs(out, exist_ok=True)
    stages = a.stages.split(",")
    width, height = (int(v) for v in a.size.split("x"))
    open(os.path.join(out, "prompt.txt"), "w").write(a.prompt)
    text, ids, mask = template(a.hf, a.prompt)
    open(os.path.join(out, "text.txt"), "w").write(text)
    save(out, "ids", ids[0][mask[0]].to(torch.int32))
    t0 = time.time()

    cap_path = os.path.join(out, "te_h35.npy")
    if "te" in stages:
        te = load_te(a.te, a.hf)
        hs = te(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        for k in (1, 18, 35, 36):
            save(out, f"te_h{k}", hs[k][0][mask[0]])
        # the pipeline's own choice, hidden_states[-2]
        assert torch.equal(hs[-2], hs[35]) and len(hs) == 37
        del te, hs
        print(f"te in {time.time() - t0:.0f} s", flush=True)
    cap = torch.from_numpy(np.load(cap_path))

    # the pipeline's latents, levels and loop (pipeline_z_image.py __call__, guidance 0)
    vae_scale = 16
    h, w = 2 * (height // vae_scale), 2 * (width // vae_scale)
    latents = torch.randn((1, 16, h, w), generator=torch.Generator("cpu").manual_seed(a.seed), dtype=torch.float32)
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(a.hf, "scheduler"))
    mu = calculate_shift((h // 2) * (w // 2), sch.config.get("base_image_seq_len", 256), sch.config.get("max_image_seq_len", 4096),
                         sch.config.get("base_shift", 0.5), sch.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(sch, a.steps, "cpu", sigmas=get_default_z_image_sigmas(a.steps), mu=mu)
    sch.set_begin_index(0)
    save(out, "sigmas", sch.sigmas)
    save(out, "lat_0", latents[0])

    if "dit" in stages:
        dit = load_dit(a.dit, a.hf)
        grabs = {}
        hooks = [(dit.t_embedder, "t_emb"), (dit.noise_refiner[-1], "nr_out"), (dit.context_refiner[-1], "cr_out"), (dit.layers[0], "l0_out"),
                 (dit.layers[-1], "l29_out")]
        hs_ = [mod.register_forward_hook(lambda _m, _i, o, n=n: grabs.__setitem__(n, o)) for mod, n in hooks]
        # the refiners' inputs: the embeddings with the pad token in the padding
        hs_ += [mod.register_forward_pre_hook(lambda _m, inp, n=n: grabs.__setitem__(n, inp[0]))
                for mod, n in [(dit.noise_refiner[0], "x_emb"), (dit.context_refiner[0], "cap_emb")]]
        print(f"dit loaded at {time.time() - t0:.0f} s", flush=True)
        for i, t in enumerate(timesteps):
            timestep = (1000 - t.expand(1)) / 1000
            x_in = [latents[0].unsqueeze(1)]
            grabs.clear()
            STATS = {} if i == 0 else None
            vel = dit(x_in, timestep, [cap], return_dict=False)[0][0].float().squeeze(1)
            save(out, f"lat_{i}", latents[0])
            save(out, f"vel_{i}", vel)
            if i == 0:
                json.dump(STATS, open(os.path.join(out, "stats.json"), "w"), indent=0)
                for n in ("t_emb", "x_emb", "nr_out", "cap_emb", "cr_out", "l0_out", "l29_out"):
                    save(out, n, grabs[n][0])
            latents = sch.step(-vel.unsqueeze(0), t, latents, return_dict=False)[0]
            print(f"step {i} (t {t.item():.2f}) at {time.time() - t0:.0f} s", flush=True)
        save(out, "lat_final", latents[0])
        for hk in hs_:
            hk.remove()
        del dit

    if "vae" in stages:
        lat = torch.from_numpy(np.load(os.path.join(out, "lat_final.npy"))).unsqueeze(0)
        vae = AutoencoderKL.from_pretrained(os.path.join(a.hf, "vae")).float().eval()
        z = lat / vae.config.scaling_factor + vae.config.shift_factor
        rgb = vae.decode(z, return_dict=False)[0][0]
        save(out, "rgb", rgb)
        from PIL import Image
        px = ((rgb.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(px).save(os.path.join(out, "ref.png"))
        print(f"vae done at {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
