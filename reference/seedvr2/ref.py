"""SeedVR2 (ByteDance-Seed, Apache-2.0) on the CPU in float32: the reference nextsycl's engine (image/seedvr2) is
checked against, stage by stage. The official code (github.com/ByteDance-Seed/SeedVR, a copy outside this
directory: --seedvr) builds the models from its own configs; the weights are numz's fp16 safetensors (the files the
model card downloads), upcast. Its CUDA-only parts are stood in for: apex's fused RMS norm by torch's, flash-attn's
varlen attention by SDPA a sequence at a time.

    python -I ref.py --seedvr DIR --dit seedvr2_ema_3b_fp16.safetensors \
        --vae ema_vae_fp16.safetensors --emb pos_emb.pt --clip CLIP.mp4|FRAMES_DIR --frames 9 --size 384x224 --out DIR

writes (float32 .npy, channels last):
    input       [T, H, W, 3]   the clip resized (bicubic) to --size, in [-1, 1]: what the engine is fed
    enc_mean    [t, h, w, 16]  the VAE encoder's posterior mean; enc_std its standard deviation
    cond        [t, h, w, 16]  the condition latent: mean * 0.9152
    noise       [t, h, w, 16]  the starting noise (seeded)
    txt         [L, 5120]      the positive text embedding (pos_emb.pt)
    blk_N       [t*h/2*w/2, D] the video tokens after DiT block N; txt_N the text tokens
    v           [t, h, w, 16]  the DiT's output (one call, t = 1000)
    x0          [t, h, w, 16]  the sampler's result (Euler, v_lerp)
    dec         [T, H, W, 3]   the VAE decode of x0 / 0.9152
    out         [T, H, W, 3]   dec after the wavelet colour fix against input
    windows.json               the window slices of both window methods for the shapes the engine checks
"""
import argparse
import json
import math
import os
import subprocess
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--seedvr", required=True)
ap.add_argument("--dit", required=True)
ap.add_argument("--vae", required=True)
ap.add_argument("--emb", required=True)
ap.add_argument("--clip", required=True)
ap.add_argument("--frames", type=int, default=9)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--size", default="384x224")
ap.add_argument("--variant", default="3b", choices=["3b", "7b"])
ap.add_argument("--seed", type=int, default=666)
ap.add_argument("--dump-blocks", default="0,1,15,31")
ap.add_argument("--autocast", action="store_true", help="the DiT under bfloat16 autocast, as the official inference runs it")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed)
torch.set_num_threads(os.cpu_count() or 8)


def save(name, t):
    np.save(os.path.join(a.out, name + ".npy"), t.detach().float().cpu().numpy().astype(np.float32))
    print(f"  {name} {tuple(t.shape)}", flush=True)


# ---- stand-ins for the CUDA-only modules -----------------------------------------------------------------------
import diffusers  # noqa: E402,F401  (before the stand-ins: it probes for flash_attn's package)

apex =types.ModuleType("apex")
apex_norm = types.ModuleType("apex.normalization")


class FusedRMSNorm(torch.nn.RMSNorm):
    def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


class FusedLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


apex_norm.FusedRMSNorm, apex_norm.FusedLayerNorm = FusedRMSNorm, FusedLayerNorm
apex.normalization = apex_norm
sys.modules["apex"], sys.modules["apex.normalization"] = apex, apex_norm

flash = types.ModuleType("flash_attn")


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, **_):
    out = torch.empty_like(q, dtype=torch.float32)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qs, ks, vs = (t.float().transpose(0, 1) for t in (q[cq[i]:cq[i + 1]], k[ck[i]:ck[i + 1]], v[ck[i]:ck[i + 1]]))
        out[cq[i]:cq[i + 1]] = F.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1)
    return out


flash.flash_attn_varlen_func = flash_attn_varlen_func
sys.modules["flash_attn"] = flash

# the causal convolutions ask for their sequence-parallel neighbours: a group of one
torch.distributed.init_process_group("gloo", init_method="tcp://127.0.0.1:29571", rank=0, world_size=1)
sys.path.insert(0, os.path.abspath(a.seedvr))
import common.distributed.basic as basic  # noqa: E402

basic.get_device = lambda: torch.device("cpu")
import common.distributed as cd  # noqa: E402

cd.get_device = basic.get_device
from common.config import create_object, load_config  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

cwd = os.getcwd()
os.chdir(a.seedvr)  # the configs inherit by paths relative to the repository
config = load_config(os.path.join(a.seedvr, f"configs_{a.variant}", "main.yaml"))
os.chdir(cwd)
import projects.video_diffusion_sr.infer as infer  # noqa: E402

infer.get_device = basic.get_device


# the wavelet colour fix as numz's ComfyUI-SeedVR2_VideoUpscaler has it (src/utils/color_fix.py, Apache-2.0): the
# result's details over the input's colours, five levels of a dilated 3 x 3 blur (replicated borders)
def wavelet_blur(image, radius):
    radius = min(radius, max(1, min(image.shape[-2:]) // 8))
    k = torch.tensor([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]], dtype=image.dtype)
    k = k[None, None].repeat(image.shape[1], 1, 1, 1)
    return F.conv2d(F.pad(image, (radius,) * 4, mode="replicate"), k, groups=image.shape[1], dilation=radius)


def wavelet_decomposition(image, levels=5):
    high = torch.zeros_like(image)
    for i in range(levels):
        low = wavelet_blur(image, 2 ** i)
        high += image - low
        image = low
    return high, low


def wavelet_reconstruction(content, style):
    return (wavelet_decomposition(content)[0] + wavelet_decomposition(style)[1]).clamp(-1, 1)


# ---- the windows: both methods' slices for the shapes the engine's tests use ------------------------------------
from models.dit_v2.window import make_720Pwindows_bysize, make_shifted_720Pwindows_bysize  # noqa: E402

win = {}
for shape in [(3, 14, 24), (6, 60, 108), (1, 60, 108), (9, 45, 80), (2, 37, 50)]:
    win[str(shape)] = {name: [[(s.start, s.stop) for s in sl] for sl in fn(shape, (4, 3, 3))]
                       for name, fn in (("win", make_720Pwindows_bysize), ("swin", make_shifted_720Pwindows_bysize))}
json.dump(win, open(os.path.join(a.out, "windows.json"), "w"))

# ---- the input: the clip's frames, bicubic to the size, [-1, 1] -------------------------------------------------
W, H = (int(v) for v in a.size.split("x"))
if os.path.isdir(a.clip):
    # a directory of the clip's frames as PNG, in name order (a host whose ffmpeg cannot decode H.264)
    from PIL import Image

    names = sorted(n for n in os.listdir(a.clip) if n.endswith(".png"))[a.start: a.start + a.frames]
    frames = torch.from_numpy(np.stack([np.asarray(Image.open(os.path.join(a.clip, n)).convert("RGB")) for n in names]))
else:
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                                "-of", "json", a.clip]))["streams"][0]
    sw, sh = probe["width"], probe["height"]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", a.clip, "-vf", f"select=gte(n\\,{a.start})", "-vsync", "0", "-frames:v", str(a.frames),
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    frames = torch.from_numpy(np.frombuffer(raw, np.uint8).copy()).view(-1, sh, sw, 3)[: a.frames]
from torchvision.transforms import functional as TVF  # noqa: E402

vid = frames.permute(0, 3, 1, 2).float() / 255.0
vid = TVF.resize(vid, size=(H, W), interpolation=TVF.InterpolationMode.BICUBIC).clamp(0, 1) * 2 - 1  # [T, 3, H, W]
save("input", vid.permute(0, 2, 3, 1))

# ---- the VAE --------------------------------------------------------------------------------------------------
vae = create_object(config.vae.model).eval()
vs = {k: v.float() for k, v in load_file(a.vae).items()}
print("vae:", vae.load_state_dict(vs, strict=False))
vae.set_causal_slicing(**config.vae.slicing)
vae.set_memory_limit(**config.vae.memory_limit)
scale = config.vae.scaling_factor
with torch.no_grad():
    x = vid.permute(1, 0, 2, 3)[None]  # [1, 3, T, H, W]
    p = vae.encode(x).posterior
    mean = p.mean[0].permute(1, 2, 3, 0)  # [t, h, w, 16]
    save("enc_mean", mean)
    save("enc_std", p.std[0].permute(1, 2, 3, 0))
cond = mean * scale
save("cond", cond)

# ---- the DiT, one call at t = 1000 ---------------------------------------------------------------------------
# made on the meta device and given the weights one at a time in float32 (the 7B's are 33 GB: never twice in memory)
with torch.device("meta"):
    dit = create_object(config.dit.model).eval()
from safetensors import safe_open  # noqa: E402

ds = {}
with safe_open(a.dit, framework="pt") as f:
    for k in f.keys():
        if not k.endswith("rope.freqs"):  # the frequencies: the model's own in float32 (the file's are fp16-rounded)
            ds[k] = f.get_tensor(k).float()
print("dit:", dit.load_state_dict(ds, strict=False, assign=True))
del ds
# rotary_embedding_torch's frequencies, as the model makes them: the 3B's "lang" (theta 10000, 42 features an axis),
# the 7B's "pixel" (linspace(1, 128, 10) * pi)
for name, m in dit.named_modules():
    if name.endswith("rope.rope"):
        n = m.freqs.shape[0]
        m.to_empty(device="cpu")  # its caches (cached_freqs, dummy) start empty
        for b in m.buffers():
            b.zero_()
        fr = (1.0 / (10000 ** (torch.arange(0, 2 * n, 2)[:n].float() / (2 * n))) if a.variant == "3b"
              else torch.linspace(1.0, 256 / 2, n) * math.pi)
        m.register_buffer("freqs", fr, persistent=False)
left = [n for n, t in list(dit.named_parameters()) + list(dit.named_buffers()) if t.is_meta]
assert not left, f"not loaded: {left[:5]}"
txt = torch.load(a.emb, weights_only=True).float()
save("txt", txt)
noise = torch.randn_like(cond)
save("noise", noise)
blocks = [int(b) for b in a.dump_blocks.split(",") if b]


def hook(i):
    def f(_m, _inp, out):
        save(f"blk_{i}", out[0])
        save(f"txt_{i}", out[1])
    return f


for i in blocks:
    if i < len(dit.blocks):
        dit.blocks[i].register_forward_hook(hook(i))
runner = infer.VideoDiffusionInfer(config)
runner.config.diffusion.cfg.scale = 1.0
runner.config.diffusion.cfg.rescale = 0.0
runner.config.diffusion.timesteps.sampling.steps = 1
runner.configure_diffusion()
t_, h_, w_, _ = cond.shape
c = torch.cat([cond, torch.ones(t_, h_, w_, 1)], -1)  # get_condition(task="sr")
shape = torch.tensor([[t_, h_, w_]])
calls = []


def model(args):
    calls.append(float(args.t.reshape(-1)[0]))
    with torch.no_grad(), torch.autocast("cpu", torch.bfloat16, enabled=a.autocast):
        v = dit(vid=torch.cat([args.x_t, c.reshape(-1, 17)], -1), txt=txt, vid_shape=shape, txt_shape=torch.tensor([[txt.shape[0]]]),
                timestep=args.t.repeat(1)).vid_sample
    save("v", v.reshape(t_, h_, w_, -1))
    return v


with torch.no_grad():
    x0 = runner.sampler.sample(x=noise.reshape(-1, 16), f=model).reshape(t_, h_, w_, 16)
print("the DiT ran at t =", calls)
save("x0", x0)

# ---- the decode and the colour fix ------------------------------------------------------------------------------
with torch.no_grad():
    dec = vae.decode((x0 / scale).permute(3, 0, 1, 2)[None]).sample[0]  # [3, T, H, W]
dec = dec.permute(1, 0, 2, 3)[: vid.shape[0]]  # [T, 3, H, W]
save("dec", dec.permute(0, 2, 3, 1))
out = wavelet_reconstruction(dec, vid)
save("out", out.permute(0, 2, 3, 1))
psnr = lambda x, y: 10 * math.log10(4.0 / float(((x - y) ** 2).mean()))  # noqa: E731
print(f"done: out vs input {psnr(out, vid):.2f} dB")
