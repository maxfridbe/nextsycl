"""MOSS-Image-SFT on the CPU in float32 - the reference for nextsycl's engine (image/mossimage): the stages' tensors
dumped for `nextsycl image check`. The DiT is OpenMOSS's own class (pipeline_moss_image.py, beside this file) with the
GGUF's weights dequantized into it, so both sides run the same numbers; the text encoder is transformers' Qwen3_5Model
on the original bf16 shards (the engine runs the Q8_0 file: a tolerance, not equality); the VAE is diffusers'
AutoencoderKLFlux2.

    python -I ref.py --dit DIT.gguf --te TE_DIR --vae VAE_DIR --tok TOK_DIR --out DIR [--prompt TEXT] [--size 512]
                     [--steps 12] [--cfg 3] [--seed 7] [--stages te,dit,vae]

Dumps (DIR, .npy float32 unless said): ids.npy (int32: the chat template's tokens); prompt.txt; te_final [T, 4096] (the
text encoder's last hidden state, after its final norm); sigmas [n + 1]; lat_0 [128, h, w] (the starting noise,
torch.randn fp32 from the seed); per step in {0, n // 2, n - 1}: lat_i (the input), vel_c_i (the conditional
prediction), vel_u_i (the negative branch with block 22 skipped), pred_i (after CFG); at step 0 the conditional
forward's: emb [N, 3072] (video_embedder), txt [T, 3072] (text_embedder), blk_B [N + T, 3072] for B in 0, 1, 21, 22, 41
(image tokens first); lat_final [128, h, w]; z [32, 2h, 2w] (the VAE's input); rgb [3, H, W] in [-1, 1]; ref.png.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_grad_enabled(False)
HERE = os.path.dirname(os.path.abspath(__file__))


def save(out, name, t):
    a = t.detach().float().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t)
    np.save(os.path.join(out, name + ".npy"), a)
    print(f"  {name} {list(a.shape)} mean {a.mean():.5f} std {a.std():.5f}", flush=True)


def tokenize(tok_dir, prompt):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"]
    return text, ids


def stage_te(a):
    from transformers import AutoModel
    text, ids = tokenize(a.tok, a.prompt)
    print(repr(text))
    open(os.path.join(a.out, "prompt.txt"), "w").write(text)
    save(a.out, "ids", ids[0].to(torch.int32))
    m = AutoModel.from_pretrained(a.te, dtype=torch.float32).eval()
    t = time.time()
    h = m(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0]
    print(f"te {time.time() - t:.1f}s")
    save(a.out, "te_final", h)
    # the negative prompt (empty) is encoded the same way: CFG's unconditional branch reads it
    ntext, nids = tokenize(a.tok, "")
    save(a.out, "ids_neg", nids[0].to(torch.int32))
    save(a.out, "te_neg", m(input_ids=nids, use_cache=False, return_dict=True).last_hidden_state[0])


def load_dit(a):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", os.path.join(HERE, "pipeline_moss_image.py"))
    P = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_moss_image"] = P
    spec.loader.exec_module(P)
    model = P.MOSSImageTransformer()
    sd = {}
    for t in GGUFReader(a.dit).tensors:
        sd[t.name] = torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(
            tuple(int(x) for x in t.shape[::-1])))
    # the file's ne = [in, out] lists the fastest dimension first: row-major that is torch's (out, in)
    print(model.load_state_dict(sd, strict=True))
    return P, model.float().eval()


def stage_dit(a):
    P, model = load_dit(a)
    te = torch.from_numpy(np.load(os.path.join(a.out, "te_final.npy")))[None]
    neg = torch.from_numpy(np.load(os.path.join(a.out, "te_neg.npy")))[None] if a.cfg != 1 and os.path.exists(os.path.join(a.out, "te_neg.npy")) else None
    n = a.size // 16
    g = torch.Generator().manual_seed(a.seed)
    lat = torch.randn((1, 128, n, n), generator=g, dtype=torch.float32)
    save(a.out, "lat_0", lat[0])
    sch = P.MOSSImageScheduler()
    sch.set_timesteps(a.steps)
    ts = sch.timesteps
    save(a.out, "sigmas", sch.sigmas)
    print("timesteps", ts.tolist()[:4], "...", len(ts))
    # the negative prompt is "" : its encoding is a second text-encoder pass (te_neg.npy, made by --stages te with --prompt "")
    mask = torch.ones(1, te.shape[1], dtype=torch.bool)
    keep = {}
    def hook(name):
        def f(mod, i, o):
            keep[name] = o[0].clone()
        return f
    hs = [model.video_embedder.register_forward_hook(hook("emb"))]
    hs.append(model.text_embedder.register_forward_hook(hook("txt")))
    for b in (0, 1, 21, 22, 41):
        hs.append(model.layers[b].register_forward_hook(hook(f"blk_{b}")))
    show = {0, len(ts) // 2, len(ts) - 1}
    for i, t in enumerate(ts):
        t0 = time.time()
        if i == 0:
            vc = model(lat, t, te, mask).sample.float()
            for k, v in keep.items():
                save(a.out, k, v)
            for h in hs:
                h.remove()
        else:
            vc = model(lat, t, te, mask).sample.float()
        pred = vc
        if a.cfg != 1.0:
            nm = neg if neg is not None else te
            vu = model(lat, t, nm, torch.ones(1, nm.shape[1], dtype=torch.bool), skip_layers=(22,)).sample.float()
            pred = vu + a.cfg * (vc - vu)
        if i in show:
            save(a.out, f"lat_{i}", lat[0])
            save(a.out, f"vel_c_{i}", vc[0])
            if a.cfg != 1.0:
                save(a.out, f"vel_u_{i}", vu[0])
            save(a.out, f"pred_{i}", pred[0])
        lat = sch.step(pred, t, lat, return_dict=False)[0]
        print(f"step {i + 1}/{len(ts)} {time.time() - t0:.1f}s", flush=True)
    save(a.out, "lat_final", lat[0])


def stage_vae(a):
    from diffusers import AutoencoderKLFlux2
    from PIL import Image
    vae = AutoencoderKLFlux2.from_pretrained(a.vae, torch_dtype=torch.float32).eval()
    lat = torch.from_numpy(np.load(os.path.join(a.out, "lat_final.npy")))[None]
    bn = vae.bn
    z = lat * (bn.running_var + vae.config.batch_norm_eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    save(a.out, "z", z[0])
    t = time.time()
    img = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    print(f"vae {time.time() - t:.1f}s")
    save(a.out, "rgb", img[0])
    px = ((img[0].permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
    Image.fromarray(px).save(os.path.join(a.out, "ref.png"))



def stage_probe(a):
    """one forward at step 0, block 0 taken apart: the modulation, the norms, q k v, attention, each residual"""
    P, model = load_dit(a)
    te = torch.from_numpy(np.load(os.path.join(a.out, "te_final.npy")))[None]
    lat = torch.from_numpy(np.load(os.path.join(a.out, "lat_0.npy")))[None]
    n = lat.shape[-1]
    t = torch.tensor([1000.0])
    with torch.no_grad():
        half = 128
        freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
        phase = t.double()[:, None] * freq
        emb = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = model.time_embedding(emb)
        mod = model.shared_block_modulation(time).view(1, 6, -1)
        head = model.video_head_modulation(time).view(1, 2, -1)
        save(a.out, "p_time", time[0]); save(a.out, "p_mod", mod[0]); save(a.out, "p_head", head[0])
        video = model.video_embedder(lat.flatten(2).transpose(1, 2).contiguous())
        rope = model.rope(n, n, te.shape[1], "cpu")
        save(a.out, "p_rope", rope[0])
        text = model.text_embedder(te, rope[:, n * n:])
        x = torch.cat((video, text), 1)
        save(a.out, "p_x0", x[0])
        blk = model.layers[0]
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.unbind(1)
        y = blk.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        save(a.out, "p_y1", y[0])
        q = blk.q_norm(blk.proj["q"](y).unflatten(-1, (24, -1))).flatten(2)
        k = blk.k_norm(blk.proj["k"](y).unflatten(-1, (24, -1))).flatten(2)
        v = blk.proj["v"](y)
        save(a.out, "p_q", q[0]); save(a.out, "p_k", k[0]); save(a.out, "p_v", v[0])
        att = P._attention(q, k, v, rope, 24)
        save(a.out, "p_att", att[0])
        x1 = x + blk.proj["o"](att) * gate_a[:, None]
        save(a.out, "p_x1", x1[0])
        y2 = blk.norm2(x1) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = blk.proj["ffn_up_gate"](y2).chunk(2, -1)
        act = P._situ(hidden, gate)
        save(a.out, "p_act", act[0])
        x2 = x1 + blk.proj["ffn_down"](act) * gate_f[:, None]
        save(a.out, "p_x2", x2[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit"); ap.add_argument("--te"); ap.add_argument("--vae"); ap.add_argument("--tok")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="A small red fox in fresh snow, soft morning light")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stages", default="te,dit,vae")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.stages.split(","):
        print("== stage", s, flush=True)
        {"te": stage_te, "dit": stage_dit, "vae": stage_vae, "probe": stage_probe}[s](a)


if __name__ == "__main__":
    main()
