"""MiniMax-H3-Image on the CPU in float32 - the reference for nextsycl's engine (image/h3image): every stage's tensors
dumped for `nextsycl image check`. Written from ComfyUI's code (comfyanonymous/ComfyUI dba377b97b17c7944ce0cf3d2429ee2caa5c4569:
comfy/ldm/minimax/model.py MiniMaxH3ImageModel, FinalLayer, _pdd_head, PackedLayout; comfy/ldm/minimax/sampling.py;
comfy/ldm/minimax/vae.py MiniMaxH3ImageVAE, ViT3DDecoder, tiled_decode; comfy/k_diffusion sample_euler) - read, not
imported: no downloaded code runs here. The weights are Comfy-Org/MiniMax-H3-Image's bf16 files (revision 892de7cb),
read a block at a time and widened to float32 for its step.

The text conditioning (H3's Qwen3-VL 32B: the video engine's, checked in its own port) comes from the engine's
`check` stage `te` (cond.npy [T, 5120], tags.npy), so this script needs no text encoder.

    python -I ref.py --dit BF16.safetensors --vae VAE.safetensors --dir DIR [--size 576x384] [--seed 3] [--steps 6]
                     [--stages dit,vae] [--threads N]

Dumps (DIR, .npy float32 unless said):
  dit:  pos [S, 3] (float64), rows [S] (int32: each token's table row at step 0), sigmas [steps + 1], noise [24, h, w],
        txt [T, 5376] (the token refiner's output), x_in [S, 5376] (the stream before block 0 at step 0), blk_B (after
        block B at step 0, B in 0, 1, 24, 49), head_I (the six heads' weights of step I), lat_I [24, h, w] (the latents
        step I starts from), vel_I [24, h, w] (the model's output: the negated head), lat_final
  vae:  raw [3, H, W] (the decoder's output, ImageNet-normalized), rgb [3, H, W] in [0, 1]
"""
import argparse
import json
import math
import os
import struct
import time

import numpy as np
import torch

torch.set_grad_enabled(False)

PDD_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
DUMP_BLOCKS = (0, 1, 24, 49)
HIDDEN, HEADS, HEAD_DIM, EPS = 5376, 56, 128, 1e-5
VISUAL_COND_TIMESTEP = 0.999
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def save(out, name, t):
    a = t.detach().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t)
    if a.dtype in (np.float16, np.float64) and name != "pos":
        a = a.astype(np.float32)
    if a.dtype == np.int64:
        a = a.astype(np.int32)
    np.save(os.path.join(out, name + ".npy"), a)
    s = a.astype(np.float64)
    print(f"  {name} {list(a.shape)} {a.dtype} mean {s.mean():.5f} std {s.std():.5f}", flush=True)


class SafeTensors:
    """A .safetensors file's header, tensors read on demand (bf16 / f16 / f32 -> float32)"""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def __contains__(self, k):
        return k in self.header

    def get(self, k):
        e = self.header[k]
        a, b = e["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            raw = f.read(b - a)
        dt = e["dtype"]
        if dt == "BF16":
            t = torch.frombuffer(bytearray(raw), dtype=torch.int16).to(torch.int32).bitwise_left_shift(16).view(torch.float32)
        elif dt == "F16":
            t = torch.frombuffer(bytearray(raw), dtype=torch.float16).float()
        elif dt == "F32":
            t = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
        else:
            raise ValueError(f"{k}: {dt}")
        return t.reshape(e["shape"])


def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_split_half(x, ang):
    # x [S, H, D]; ang [S, rot/2]: features i and i + rot/2 rotate together by ang[:, i] (comfy_kitchen's split half)
    r2 = ang.shape[1]
    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., :r2], x[..., r2:2 * r2]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., 2 * r2:]], dim=-1)


def attention(q, k, v):
    # [S, H, D] -> [S, H * D]
    o = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
    return o.transpose(0, 1).reshape(q.shape[0], -1)


def swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def axis(dim, sqrt_area):
    ratio = dim / sqrt_area
    n = dim // 2
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def layout(text_len, lh, lw):
    """PackedLayout for text to image (no references, no audio): [S, 3] float64 positions and the segments"""
    area = math.sqrt(lh * lw)
    hh, ww = torch.meshgrid(axis(lh, area), axis(lw, area), indexing="ij")
    pos = torch.zeros(text_len, 3, dtype=torch.float64)
    pos[:, 0] = torch.arange(text_len, dtype=torch.float64)
    img = torch.empty(hh.numel(), 3, dtype=torch.float64)
    img[:, 0] = float(text_len)
    img[:, 1] = hh.reshape(-1)
    img[:, 2] = ww.reshape(-1)
    return torch.cat([pos, img]), [("text", 0, text_len), ("video", text_len, text_len + img.shape[0])]


def patchify(lat):
    # [24, h, w] -> [h/2 * w/2, 96], features (c, dy, dx)
    c, h, w = lat.shape
    return lat.reshape(c, h // 2, 2, w // 2, 2).permute(1, 3, 0, 2, 4).reshape(-1, c * 4)


def unpatchify(rows, h, w):
    return rows.reshape(h // 2, w // 2, 24, 2, 2).permute(2, 0, 3, 1, 4).reshape(24, h, w)


class Dit:
    def __init__(self, path, threads):
        self.st = SafeTensors(path)
        self.n = 1 + max(int(k.split(".")[1]) for k in self.st.header if k.startswith("blocks."))
        g = self.st.get
        self.table = g("adaln_t_table")
        self.inv_freq = g("rope.inv_freq")
        self.patch_w, self.patch_b = g("video_patch_proj.weight"), g("video_patch_proj.bias")
        self.fin_norm = g("final_layer.norm.weight")
        self.fin_w, self.fin_b = g("final_layer.adaln_proj.linear.weight"), g("final_layer.adaln_proj.linear.bias")
        self.out_w, self.out_b = g("final_layer.video_out.weight"), g("final_layer.video_out.bias")
        torch.set_num_threads(threads)

    def block(self, p):
        g = self.st.get
        return {k: g(f"{p}.{k}") for k in ("norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight", "attn.qkv_proj.weight",
                                          "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")} | (
            {"adaln.w": g(f"{p}.adaln_proj.linear.weight"), "adaln.b": g(f"{p}.adaln_proj.linear.bias")} if f"{p}.adaln_proj.linear.weight" in self.st else {})

    def refine(self, cond):
        g = self.st.get
        x = cond @ g("condition_proj.weight").T + g("condition_proj.bias")
        nref = 1 + max(int(k.split(".")[2]) for k in self.st.header if k.startswith("token_refiner.blocks."))
        for i in range(nref):
            b = self.block(f"token_refiner.blocks.{i}")
            q, k, v = (rms(x, b["norm1.weight"]) @ b["attn.qkv_proj.weight"].T).split(HEADS * HEAD_DIM, dim=-1)
            s = x.shape[0]
            q = rms(q.view(s, HEADS, HEAD_DIM), b["attn.q_norm.weight"])
            k = rms(k.view(s, HEADS, HEAD_DIM), b["attn.k_norm.weight"])
            x = attention(q, k, v.view(s, HEADS, HEAD_DIM)) @ b["attn.out_proj.weight"].T + x
            x = swiglu(rms(x, b["norm2.weight"]) @ b["mlp.fc1.weight"].T) @ b["mlp.fc2.weight"].T + x
        return rms(x, g("token_refiner.final_norm.weight"))

    def t_emb(self, unique_t):
        t = torch.tensor(unique_t, dtype=torch.float32)
        pos = t.clamp(0.0, 1.0) * (self.table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=self.table.shape[0] - 2)
        return torch.lerp(self.table[i0], self.table[i0 + 1], (pos - i0).unsqueeze(1))

    def forward(self, lat, txt, tags, sigma_in, sample_sigmas, dump=None, ang=None):
        """lat [24, h, w]; returns the model's output (the negated head) [24, h, w] and the rows"""
        _, h, w = lat.shape
        tl = txt.shape[0]
        sigma = float(torch.tensor(sigma_in * 1000.0, dtype=torch.float32)) / 1000.0
        t = 1.0 - sigma
        unique_t = sorted({t})
        t_row = {v: i for i, v in enumerate(unique_t)}
        rows = torch.cat([t_row[t] * 3 + tags.long(), torch.full(((h // 2) * (w // 2),), t_row[t] * 3 + 0, dtype=torch.long)])
        te = self.t_emb(unique_t)
        x = torch.cat([txt, patchify(lat) @ self.patch_w.T + self.patch_b])
        if dump is not None:
            dump("x_in", x)
            dump("rows", rows)
        s = x.shape[0]
        for i in range(self.n):
            b = self.block(f"blocks.{i}")
            tables = (te @ b["adaln.w"].T + b["adaln.b"]).view(te.shape[0] * 3, 6 * HIDDEN).chunk(6, dim=-1)
            sh_a, sc_a, g_a, sh_m, sc_m, g_m = (tb[rows] for tb in tables)
            hcur = rms(x, b["norm1.weight"]) * (1.0 + sc_a) + sh_a
            q, k, v = (hcur @ b["attn.qkv_proj.weight"].T).split(HEADS * HEAD_DIM, dim=-1)
            q = rope_split_half(rms(q.view(s, HEADS, HEAD_DIM), b["attn.q_norm.weight"]), ang)
            k = rope_split_half(rms(k.view(s, HEADS, HEAD_DIM), b["attn.k_norm.weight"]), ang)
            x = x + (attention(q, k, v.view(s, HEADS, HEAD_DIM)) @ b["attn.out_proj.weight"].T) * g_a
            hcur = rms(x, b["norm2.weight"]) * (1.0 + sc_m) + sh_m
            x = x + (swiglu(hcur @ b["mlp.fc1.weight"].T) @ b["mlp.fc2.weight"].T) * g_m
            if dump is not None and i in DUMP_BLOCKS:
                dump(f"blk_{i}", x)
        shift, scale = (te @ self.fin_w.T + self.fin_b).chunk(2, dim=-1)
        r = t_row[t]
        hfin = rms(x[tl:], self.fin_norm) * (1.0 + scale[r]) + shift[r]
        # the PDD bank: the step from sigma to the schedule's next point takes the heads of the intervals it spans
        n = self.out_w.shape[0] // 96
        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        start, stop = (min(range(n + 1), key=lambda kk: abs(PDD_SIGMAS[kk] - float(sv))) for sv in (sigma, sigma_next))
        start = min(start, n - 1)
        stop = max(stop, start + 1)
        dt = [PDD_SIGMAS[kk] - PDD_SIGMAS[kk + 1] for kk in range(start, stop)]
        wts = torch.tensor(dt, dtype=torch.float32) / sum(dt)
        wrows = self.out_w.reshape(n, -1, self.out_w.shape[1])
        brows = self.out_b.reshape(n, -1)
        first = max(start, 1)
        wsum = wrows[0] + torch.einsum("n,noi->oi", wts[first - start:], wrows[first:stop])
        bsum = brows[0] + torch.einsum("n,no->o", wts[first - start:], brows[first:stop])
        head = [0.0] * n
        head[0] = 1.0
        for kk in range(first, stop):
            head[kk] = float(wts[kk - start])
        v = hfin @ wsum.T + bsum
        return -unpatchify(v, h, w), head


def rope_angles(pos, inv_freq):
    # ComfyUI's rope_freqs: [S, 3] -> per axis x 16 frequencies, (t, h, w) one after the other: [S, 48]
    p = pos.to(torch.float32)
    per = p.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    return torch.cat(per.unbind(dim=1), dim=-1)


def run_dit(a, out):
    cond = torch.from_numpy(np.load(os.path.join(out, "cond.npy"))).float()
    tags = torch.from_numpy(np.load(os.path.join(out, "tags.npy"))).long()
    w, h = (int(v) for v in a.size.split("x"))
    lh, lw = h // 16, w // 16
    d = Dit(a.dit, a.threads)
    t0 = time.time()
    txt = d.refine(cond)
    save(out, "txt", txt)
    pos, _ = layout(txt.shape[0], lh, lw)
    save(out, "pos", pos)
    ang = rope_angles(pos, d.inv_freq)
    n = len(PDD_SIGMAS) - 1
    if a.steps == n:
        sig = PDD_SIGMAS
    else:
        sig = []
        for i in range(a.steps + 1):
            p = i / a.steps * n
            k = min(int(p), n - 1)
            sig.append(PDD_SIGMAS[k] + (p - k) * (PDD_SIGMAS[k + 1] - PDD_SIGMAS[k]))
    sigmas = torch.tensor(sig, dtype=torch.float32)
    save(out, "sigmas", sigmas)
    x = torch.randn((1, 24, 1, lh, lw), generator=torch.manual_seed(a.seed), dtype=torch.float32)[0, :, 0]
    save(out, "noise", x)
    for i in range(len(sig) - 1):
        save(out, f"lat_{i}", x)
        dump = (lambda name, t: save(out, name, t)) if i == 0 else None
        vel, head = d.forward(x, txt, tags, float(sigmas[i]), sigmas, dump=dump, ang=ang)
        save(out, f"vel_{i}", vel)
        save(out, f"head_{i}", torch.tensor(head))
        # sample_euler: denoised = x - v sigma, d = (x - denoised) / sigma, x += d (next - sigma)
        s, nx = sigmas[i], sigmas[i + 1]
        den = x - vel * s
        x = x + (x - den) / s * (nx - s)
        print(f"step {i}: {time.time() - t0:.0f} s", flush=True)
    save(out, "lat_final", x)


class Decoder:
    def __init__(self, path):
        st = SafeTensors(path)
        g = st.get
        self.n = 1 + max(int(k.split(".")[2]) for k in st.header if k.startswith("decoder.transformer_blocks."))
        self.blocks = [{k: g(f"decoder.transformer_blocks.{i}.{k}") for k in (
            "norm1.weight", "norm2.weight", "scale1", "scale2", "attn.to_qkv.weight", "attn.to_qkv.bias", "attn.to_out.weight",
            "attn.to_out.bias", "ff.w1.weight", "ff.w1.bias", "ff.w2.weight", "ff.w2.bias")} for i in range(self.n)]
        self.x_w, self.x_b = g("decoder.x_embedder.weight"), g("decoder.x_embedder.bias")
        self.reg = g("decoder.register_tokens")[0]
        self.no_w, self.no_b = g("decoder.norm_out.weight"), g("decoder.norm_out.bias")
        self.po_w, self.po_b = g("decoder.proj_out.weight"), g("decoder.proj_out.bias")
        self.pq_w, self.pq_b = g("post_quant_conv.weight")[:, :, 0, 0], g("post_quant_conv.bias")
        self.dim = self.x_w.shape[0]
        self.heads = self.dim // 64
        # RotaryEmbeddingND(dim=32, base 100, n_dim=2)
        self.inv_freq = 1 / 100.0 ** torch.arange(0, 1, 2 * 2 / 32, dtype=torch.float32)

    def tile(self, z):
        # z [24, h, w] (after the post-quant conv) -> pixels [3, 16h, 16w], ImageNet-normalized
        c, h, w = z.shape
        x = z.reshape(c, h * w).T @ self.x_w.T + self.x_b
        np_ = x.shape[0]
        x = torch.cat([x, self.reg, torch.zeros(1, self.dim)])
        ch = (torch.arange(0.5, h, dtype=torch.float32) / h) * 2.0 - 1.0
        cw = (torch.arange(0.5, w, dtype=torch.float32) / w) * 2.0 - 1.0
        ids = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).reshape(-1, 2)
        ids = torch.cat([ids, torch.zeros(5, 2)])
        ang = (2.0 * math.pi * ids[:, :, None] * self.inv_freq[None, None, :]).flatten(1, 2)  # [S, 16]
        s = x.shape[0]
        for b in self.blocks:
            qkv = rms(x, b["norm1.weight"]) @ b["attn.to_qkv.weight"].T + b["attn.to_qkv.bias"]
            qkv = qkv.view(s, self.heads, 3 * 64)
            q, k, v = qkv.chunk(3, dim=-1)
            q = rope_split_half(rms(q, torch.ones(64)), ang)
            k = rope_split_half(rms(k, torch.ones(64)), ang)
            x = x + (attention(q, k, v) @ b["attn.to_out.weight"].T + b["attn.to_out.bias"]) * b["scale1"]
            hh = swiglu(rms(x, b["norm2.weight"]) @ b["ff.w1.weight"].T + b["ff.w1.bias"])
            x = x + (hh @ b["ff.w2.weight"].T + b["ff.w2.bias"]) * b["scale2"]
        o = torch.nn.functional.layer_norm(x, (self.dim,), self.no_w, self.no_b, 1e-5) @ self.po_w.T + self.po_b
        o = o[:np_].view(h, w, 3, 16, 16)
        return o.permute(2, 0, 3, 1, 4).reshape(3, h * 16, w * 16)


def split_tiles(n, tile=256, ov=64, ratio=16):
    if tile >= n:
        return [0], [n], []
    N = math.ceil(n / tile)
    while True:
        overlaps = [ov] * (N - 1)
        rem = tile * N - sum(overlaps) - n
        if rem < 0:
            N += 1
        else:
            break
    for i in range(rem // ratio):
        overlaps[i % (N - 1)] += ratio
    idx = [0]
    for i in range(N - 1):
        idx.append(idx[-1] + tile - overlaps[i])
    return idx, [tile] * N, overlaps


def blend(a, b, e, dim):
    e = min(a.shape[dim], b.shape[dim], e)
    p = torch.arange(e, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = e
    wa, wb = (1 - p / e).view(shape), (p / e).view(shape)
    sa = [slice(None)] * a.ndim
    sa[dim] = slice(-e, None)
    sb = [slice(None)] * b.ndim
    sb[dim] = slice(0, e)
    out = a[tuple(sa)] * wa + b[tuple(sb)] * wb
    if e < b.shape[dim]:
        rest = [slice(None)] * b.ndim
        rest[dim] = slice(e, None)
        return torch.cat([out, b[tuple(rest)]], dim=dim)
    return out


def tiled_decode(dec, z):
    # ComfyUI's tiled_decode for one frame: [24, h, w] -> [3, 16h, 16w]
    H, W = z.shape[1] * 16, z.shape[2] * 16
    yi, yl, yo = split_tiles(H)
    xi, xl, xo = split_tiles(W)
    canvas = torch.empty(3, H, W)
    strip = None
    out_y = 0
    for i, (ip, il) in enumerate(zip(yi, yl)):
        new_strip, left_tail, out_x = None, None, 0
        for j, (jp, jl) in enumerate(zip(xi, xl)):
            tile = dec.tile(z[:, ip // 16:(ip + il) // 16, jp // 16:(jp + jl) // 16])
            if i > 0:
                tile = blend(strip[:, :, jp:jp + jl], tile, yo[i - 1], dim=-2)
            if j > 0:
                tile = blend(left_tail, tile, xo[j - 1], dim=-1)
            left_tail = tile[:, :, -xo[j]:].clone() if j < len(xi) - 1 else None
            if j < len(xi) - 1:
                tile = tile[:, :, :-xo[j]]
            if i < len(yi) - 1:
                if new_strip is None:
                    new_strip = torch.empty(3, yo[i], W)
                new_strip[:, :, out_x:out_x + tile.shape[-1]] = tile[:, -yo[i]:, :]
                tile = tile[:, :-yo[i], :]
            canvas[:, out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]] = tile
            th = tile.shape[-2]
            out_x += tile.shape[-1]
        strip = new_strip
        out_y += th
    return canvas


def run_vae(a, out):
    z = torch.from_numpy(np.load(os.path.join(out, "lat_final.npy"))).float()
    dec = Decoder(a.vae)
    zr = z * torch.tensor(LATENTS_STD).view(-1, 1, 1) + torch.tensor(LATENTS_MEAN).view(-1, 1, 1)
    zp = torch.einsum("oc,chw->ohw", dec.pq_w, zr) + dec.pq_b.view(-1, 1, 1)
    raw = tiled_decode(dec, zp)
    save(out, "raw", raw)
    rgb = (raw * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)).clamp(0.0, 1.0)
    save(out, "rgb", rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit")
    ap.add_argument("--vae")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--size", default="576x384")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--stages", default="dit,vae")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for s in a.stages.split(","):
        print(f"== {s}", flush=True)
        {"dit": run_dit, "vae": run_vae}[s](a, a.dir)


if __name__ == "__main__":
    main()
