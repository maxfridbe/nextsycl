"""FlashVSR v1.1 (tiny decoder) on the CPU in float32 - the reference for nextsycl's engine (image/flashvsr; nextcuda's
reference/flashvsr/ref.py, the same dumps, so either engine's check reads them): the
official pipeline (github.com/OpenImagingLab/FlashVSR @ 0d403ee3: diffsynth/pipelines/flashvsr_tiny.py's
FlashVSRTinyPipeline, models/wan_video_dit.py, examples/WanVSR/utils/{utils.py: Causal_LQ4x_Proj, TCDecoder.py})
run as it is, its modules loaded from the source tree. Two things are not the official code's:

- block_sparse_attn (mit-han-lab's CUDA kernel, not installable on the CPU) is replaced by its exact equivalent: per
  head, softmax over the keys of the 128 x 128 blocks the mask selects, a query row with no selected block giving 0
  (the kernel's 1 / sum guard);
- the input is prepared as the engine prepares it (the official script centre-crops the upscaled frames to multiples
  of 128; the engine mirrors them out to the next multiple and crops the result back): PIL bicubic to scale x the
  frame, numpy 'reflect' padding split evenly, the clip's last frame repeated to 8n + 1 frames with 4 or more past
  its end.

    python -I ref.py --src FLASHVSR_SRC --model MODEL_DIR --frames PNG_DIR --out DIR
        [--scale 2] [--seed 0] [--count N] [--sparse-ratio 2.0] [--local-range 11] [--threads N]

(hardac1, 33 frames of an H3 film at 512 x 384 -> 1024 x 768, 16 threads: see docs/ports/flashvsr.md for the run.)

MODEL_DIR holds the FlashVSR-v1.1 repository's files (diffusion_pytorch_model_streaming_dmd.safetensors,
LQ_proj_in.ckpt, TCDecoder.ckpt) and posi_prompt.pth (examples/WanVSR/prompt_tensor). Dumps (DIR, .npy float32
unless said): lq_u8 [F, H, W, 3] uint8 (the prepared frames), noise [16, F', H/8, W/8], for chunks 0 and 1:
c{i}_x (the chunk's noise), c{i}_lq [L, 1536] (the LQ projection, block 0's residual), c{i}_b0 [L, 1536] (block 0's
output), c{i}_b15, c{i}_b29, c{i}_pred [16, f, H/8, W/8] (the DiT's output); dec [3, T, H, W] (the decoder's frames in
[-1, 1], before the colour fix) and out [3, T, H, W] (after it); out_NNN.png.
"""
import argparse
import glob
import importlib.util
import math
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

torch.set_grad_enabled(False)
DUMP = {}


def block_sparse_attn_func(q, k, v, cu_q, cu_k, head_mask_type, streaming_info, base_blockmask, max_q, max_k, p_dropout,
                           deterministic=False, softmax_scale=None, is_causal=False, exact_streaming=False, return_attn_probs=False):
    """q [Sq, H, D], k / v [Sk, H, D]; base_blockmask [1, H, Sq/128, Sk/128] (every head its own: head_mask_type all 1)"""
    sq, nh, d = q.shape
    sk = k.shape[0]
    out = torch.zeros(sq, nh, d, dtype=torch.float32)
    for h in range(nh):
        m = base_blockmask[0, h].repeat_interleave(128, 0)[:sq].repeat_interleave(128, 1)[:, :sk]
        qh, kh, vh = q[:, h].float(), k[:, h].float(), v[:, h].float()
        for r0 in range(0, sq, 4096):
            s = qh[r0:r0 + 4096] @ kh.T / math.sqrt(d)
            mm = m[r0:r0 + 4096]
            s = s.masked_fill(~mm, -float("inf"))
            mx = s.amax(-1, keepdim=True)
            mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
            pr = torch.exp(s - mx)
            l = pr.sum(-1, keepdim=True)
            pr = pr / torch.where(l == 0, torch.ones_like(l), l)
            out[r0:r0 + 4096, h] = pr @ vh
    return out.to(q.dtype)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def package(name):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
    return m


def save(out, name, t):
    a = t.detach().float().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t)
    np.save(os.path.join(out, name + ".npy"), a)


def prepare(paths, scale):
    """The engine's preparation: (frames uint8 [F, H, W, 3] padded to multiples of 128, the real count, crop box)"""
    n = len(paths)
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w0, h0 = im.size
        sw, sh = int(round(w0 * scale)), int(round(h0 * scale))
        a = np.asarray(im.resize((sw, sh), Image.BICUBIC))
        tw, th = (sw + 127) // 128 * 128, (sh + 127) // 128 * 128
        l, t = (tw - sw) // 2, (th - sh) // 2
        a = np.pad(a, ((t, th - sh - t), (l, tw - sw - l), (0, 0)), mode="reflect")
        frames.append(a)
    nn = (n + 3 + 7) // 8
    f = 8 * nn + 1
    idx = list(range(n)) + [n - 1] * (f - n)
    return np.stack([frames[i] for i in idx]), n, (l, t, sw, sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--local-range", type=int, default=11)
    ap.add_argument("--sparse-ratio", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0: all; a GPU box's film jobs want some left)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads or os.cpu_count())

    bsa = types.ModuleType("block_sparse_attn")
    bsa.block_sparse_attn_func = block_sparse_attn_func
    sys.modules["block_sparse_attn"] = bsa
    src = os.path.join(a.src, "diffsynth")
    package("diffsynth")
    models = package("diffsynth.models")
    models.ModelManager = None
    load("diffsynth.models.utils", os.path.join(src, "models/utils.py"))
    dit_mod = load("diffsynth.models.wan_video_dit", os.path.join(src, "models/wan_video_dit.py"))
    # the cross-attention in exact arithmetic (scaled_dot_product_attention), not whichever fused kernel is installed
    dit_mod.FLASH_ATTN_3_AVAILABLE = dit_mod.FLASH_ATTN_2_AVAILABLE = dit_mod.SAGE_ATTN_AVAILABLE = False
    load("diffsynth.models.wan_video_vae", os.path.join(src, "models/wan_video_vae.py"))
    package("diffsynth.schedulers")
    load("diffsynth.schedulers.flow_match", os.path.join(src, "schedulers/flow_match.py"))
    package("diffsynth.pipelines")
    load("diffsynth.pipelines.base", os.path.join(src, "pipelines/base.py"))
    pipe_mod = load("diffsynth.pipelines.flashvsr_tiny", os.path.join(src, "pipelines/flashvsr_tiny.py"))
    ex = os.path.join(a.src, "examples/WanVSR/utils")
    utils = load("fv_utils", os.path.join(ex, "utils.py"))
    tcd = load("fv_tcd", os.path.join(ex, "TCDecoder.py"))

    from safetensors.torch import load_file
    dit = dit_mod.WanModel(dim=1536, in_dim=16, ffn_dim=8960, out_dim=16, text_dim=4096, freq_dim=256, eps=1e-6,
                           patch_size=(1, 2, 2), num_heads=12, num_layers=30)
    dit.load_state_dict(load_file(os.path.join(a.model, "diffusion_pytorch_model_streaming_dmd.safetensors")), strict=True)
    dit.eval()
    pipe = pipe_mod.FlashVSRTinyPipeline(device="cpu", torch_dtype=torch.float32)
    pipe.dit = dit
    pipe.dit.LQ_proj_in = utils.Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).float()
    pipe.dit.LQ_proj_in.load_state_dict(torch.load(os.path.join(a.model, "LQ_proj_in.ckpt"), map_location="cpu", weights_only=True), strict=True)
    pipe.TCDecoder = tcd.build_tcdecoder(new_channels=[512, 256, 128, 128], device="cpu", dtype=torch.float32, new_latent_channels=16 + 768)
    print(pipe.TCDecoder.load_state_dict(torch.load(os.path.join(a.model, "TCDecoder.ckpt"), map_location="cpu", weights_only=True), strict=False))
    pipe.TCDecoder.eval()
    ctx = torch.load(os.path.join(a.model, "posi_prompt.pth"), map_location="cpu", weights_only=True)
    pipe.init_cross_kv(context_tensor=ctx)

    paths = sorted(glob.glob(os.path.join(a.frames, "*.png")))
    if a.count:
        paths = paths[:a.count]
    u8, n, box = prepare(paths, a.scale)
    np.save(os.path.join(a.out, "lq_u8.npy"), u8)
    f, th, tw = u8.shape[0], u8.shape[1], u8.shape[2]
    print(f"{n} frames -> {f} prepared at {tw}x{th}, crop {box}")
    lq = torch.from_numpy(u8).float().permute(3, 0, 1, 2).unsqueeze(0) / 255.0 * 2.0 - 1.0

    # the pipeline's own noise, made as it makes it (if_buffer: (F - 1) / 4 latent frames)
    noise = torch.randn((1, 16, (f - 1) // 4, th // 8, tw // 8), generator=torch.Generator("cpu").manual_seed(a.seed), dtype=torch.float32)
    save(a.out, "noise", noise[0])

    chunk = {"i": 0}
    orig = pipe_mod.model_fn_wan_video

    def blk_hook(idx):
        def h(_m, _inp, outp):
            if chunk["i"] < 2:
                save(a.out, f"c{chunk['i']}_b{idx}", outp[0][0])
        return h
    for idx in (0, 15, 29):
        dit.blocks[idx].register_forward_hook(blk_hook(idx))

    def fn(dit_, x, *args, **kw):
        i = chunk["i"]
        if i < 2:
            save(a.out, f"c{i}_x", x[0])
            save(a.out, f"c{i}_lq", kw["LQ_latents"][0][0])
        r = orig(dit_, x, *args, **kw)
        if i < 2:
            save(a.out, f"c{i}_pred", r[0][0])
        print(f"chunk {i} done", flush=True)
        chunk["i"] += 1
        return r
    pipe_mod.model_fn_wan_video = fn

    cc = pipe.ColorCorrector.forward

    def color(hq, lq_, **kw):
        save(a.out, "dec", hq[0])
        r = cc(hq, lq_, **kw)
        save(a.out, "out", r[0])
        return r
    pipe.ColorCorrector.forward = color
    pipe.generate_noise = lambda shape, seed=None, device="cpu", dtype=None: noise.clone()

    frames = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=a.seed, LQ_video=lq, num_frames=f,
                  height=th, width=tw, is_full_block=False, if_buffer=True, topk_ratio=a.sparse_ratio * 768 * 1280 / (th * tw),
                  kv_ratio=3.0, local_range=a.local_range, color_fix=True)
    l, t, sw, sh = box
    v = ((frames.float() + 1) * 127.5).clip(0, 255).numpy().astype(np.uint8)
    for i in range(min(n, v.shape[1])):
        Image.fromarray(v[:, i, t:t + sh, l:l + sw].transpose(1, 2, 0)).save(os.path.join(a.out, f"out_{i:03d}.png"))
    print("done", v.shape)


if __name__ == "__main__":
    main()
