"""reference/trellis2/ref.py: TRELLIS.2 on the CPU through its own modules (microsoft/TRELLIS.2, fetch.sh), on the SAME
files nextsycl reads (Aero-Ex/Trellis2-GGUF: the Q8_0 flows dequantized exactly, the fp16 decoders, DINOv3, BiRefNet),
dumping every stage the engine is checked against (`nextsycl model3d check DIR --stages ...`).

    sh fetch.sh T2DIR
    python ref.py PICTURE OUT [--t2 T2DIR] [--models ~/models/trellis2] [--pipeline 512|1024_cascade] [--seed 42]
                  [--stages pre,dino,ss,shape,tex,sdec,tdec] [--steps 12] [--max-tokens 49152]

Stand-ins for the CUDA-only parts (the modules themselves are untouched):
  - the submanifold sparse convolution (flex_gemm): a gather - a [N, 27] neighbour map from sorted coordinate keys
    (tap (kd, kh, kw) reads the voxel at (x + kd - 1, y + kh - 1, z + kw - 1), the weight [Co, Kd, Kh, Kw, Ci]), 27
    products accumulated in float32 (trellis.cpp's tools/ref_sparse_conv.py convention);
  - sparse attention (flash_attn / xformers): torch SDPA, exact at batch 1; dense attention runs ATTN_BACKEND=sdpa;
  - o_voxel's CUDA hashmap in flexible_dual_grid_to_mesh: sorted keys + searchsorted, the same quads and splits.

Every stage reads what it needs from OUT, so they can run one at a time:
  pre:   pre.png (the square, premultiplied picture: alpha > 0.8 box, the background cut by BiRefNet when the picture
         is opaque - then rm_in.npy [3,1024,1024], rm_x1.npy, rm_x4.npy (the encoder's finest and coarsest
         features), rm_logits.npy [1024,1024], rm_mask.png too)
  dino:  dino512_in.npy [3,512,512] (normalized), cond512.npy [1029,1024]; cond1024.npy [4101,1024] unless --pipeline 512
  ss:    ss_noise.npy [8,16,16,16], ss_in0/ss_blk0/ss_blk29.npy (the first call's input layer, block 0 and the last
         block, cond), ss_call0.npy / ss_call1.npy (the first velocities: cond, uncond), ss_x<i>.npy after each step,
         ss_logits.npy [64,64,64], ss_coords.npy [N,3]
  shape: shape_noise.npy, shape_call0.npy, shape_x<i>.npy, shape_slat.npy (denormalized) [N,32]; the cascade also
         hr_coords.npy (after the budget), shape_hr_*.npy, info.json's res
  tex:   tex_noise.npy, tex_call0.npy, tex_slat.npy (denormalized)
  sdec:  sdec_c<i>.npy (coordinates entering up block i), sdec_sub<i>.npy (its subdivision logits), sdec_up<i>.npy
         (its output features), sdec_h.npy [M,7] and sdec_coords.npy [M,3], mesh_v.npy / mesh_f.npy
  tdec:  tdec.npy [M,6] (PBR in [0,1]: base colour, metallic, roughness, alpha)
"""
import argparse
import json
import os
import sys
import time
import types

os.environ.setdefault("ATTN_BACKEND", "sdpa")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count() or 8)))

FLOWS = {
    "ss": "refiner/ss_flow_img_dit_1_3B_64_bf16",
    "shape512": "shape/slat_flow_img2shape_dit_1_3B_512_bf16",
    "shape1024": "shape/slat_flow_img2shape_dit_1_3B_1024_bf16",
    "tex512": "texture/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    "tex1024": "texture/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
}
DECODERS = {
    "ss": "decoders/Stage1/ss_dec_conv3d_16l8_fp16",
    "shape": "decoders/Stage2/shape_dec_next_dc_f16c32_fp16",
    "tex": "decoders/Stage2/tex_dec_next_dc_f16c32_fp16",
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def save(out, name, a):
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy()
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(np.asarray(a, dtype=np.float32)))


def load(out, name):
    return torch.from_numpy(np.load(os.path.join(out, name + ".npy")))


# ---- stand-ins for the CUDA extensions --------------------------------------------------------------------------------

def neighbours(coords):
    """[N, 27] row of each voxel's neighbour (-1: none), tap t = kd * 9 + kh * 3 + kw at offset (kd-1, kh-1, kw-1)"""
    c = coords[:, 1:].long() + 1
    b = coords[:, 0].long()
    R = int(c.max()) + 2
    key = ((b * R + c[:, 0]) * R + c[:, 1]) * R + c[:, 2]
    sk, order = torch.sort(key)
    n = coords.shape[0]
    nm = torch.empty(n, 27, dtype=torch.long)
    for t in range(27):
        d0, d1, d2 = t // 9 - 1, t // 3 % 3 - 1, t % 3 - 1
        q = ((b * R + c[:, 0] + d0) * R + c[:, 1] + d1) * R + c[:, 2] + d2
        pos = torch.searchsorted(sk, q).clamp_(max=n - 1)
        nm[:, t] = torch.where(sk[pos] == q, order[pos], torch.full_like(pos, -1))
    return nm


def gather_conv(feats, nm, w, b):
    n, ci = feats.shape
    co = w.shape[0]
    wt = w.float().reshape(co, 27, ci)
    fz = torch.cat([feats.float(), feats.new_zeros(1, ci, dtype=torch.float32)])
    idx = torch.where(nm < 0, torch.full_like(nm, n), nm)
    out = (b.float()[None].expand(n, co).clone() if b is not None else torch.zeros(n, co))
    step = max(1, (1 << 27) // max(ci, co))
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        acc = out[r0:r1]
        for t in range(27):
            acc.addmm_(fz[idx[r0:r1, t]], wt[:, t, :].t())
    return out


class GatherConvBackend:
    @staticmethod
    def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        k = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, k
        self.stride, self.dilation = (1, 1, 1), (1, 1, 1)
        self.weight = nn.Parameter(torch.empty((out_channels, *k, in_channels)))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    @staticmethod
    def sparse_conv3d_forward(self, x):
        nm = x.get_spatial_cache("ref_neighbours")
        if nm is None:
            nm = neighbours(x.coords)
            x.register_spatial_cache("ref_neighbours", nm)
        return x.replace(gather_conv(x.feats, nm, self.weight, self.bias).to(x.feats.dtype))


def fdg_to_mesh(coords, dual_vertices, intersected_flag, split_weight, aabb, voxel_size=None, grid_size=None, train=False):
    """o_voxel.convert.flexible_dual_grid_to_mesh (inference), its hashmap as sorted keys"""
    aabb = torch.tensor(np.array(aabb), dtype=torch.float32)
    gs = torch.tensor([grid_size] * 3 if isinstance(grid_size, int) else grid_size, dtype=torch.float32)
    vs = (aabb[1] - aabb[0]) / gs
    off = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ], dtype=torch.long)
    c = coords.long()
    n = c.shape[0]
    R = int(c.max()) + 3
    key = lambda p: (p[..., 0] * R + p[..., 1]) * R + p[..., 2]
    sk, order = torch.sort(key(c))
    conn = (c[:, None, None, :] + off[None])[intersected_flag]          # (M, 4, 3)
    q = key(conn)
    pos = torch.searchsorted(sk, q).clamp_(max=n - 1)
    idx = torch.where(sk[pos] == q, order[pos], torch.full_like(pos, -1))
    quads = idx[(idx >= 0).all(dim=1)]
    verts = (c.float() + dual_vertices) * vs + aabb[0]
    sw = split_weight[quads]
    s02, s13 = sw[:, 0] * sw[:, 2], sw[:, 1] * sw[:, 3]
    tris = torch.where(s02 > s13, quads[:, [0, 1, 2, 0, 2, 3]], quads[:, [0, 1, 3, 3, 1, 2]]).reshape(-1, 3)
    return verts, tris


class MeshStub:
    def __init__(self, vertices, faces, *a, **k):
        self.vertices, self.faces = vertices, faces


def setup(t2):
    def stub(name, **kw):
        m = types.ModuleType(name)
        m.__dict__.update(kw)
        sys.modules[name] = m
        return m
    for n in ["cumesh", "cumesh.remeshing", "nvdiffrast", "nvdiffrast.torch", "flex_gemm", "flex_gemm.ops", "o_voxel", "o_voxel.postprocess"]:
        stub(n)
    stub("flex_gemm.ops.grid_sample", grid_sample_3d=None)
    stub("flex_gemm.ops.spconv", sparse_submanifold_conv3d=None, set_algorithm=None, set_hashmap_ratio=None)
    stub("o_voxel.convert", flexible_dual_grid_to_mesh=fdg_to_mesh)
    # the packages whose __init__ would import the whole pipeline (renderers, CUDA extensions): empty, with their path
    for p in ["trellis2", "trellis2/pipelines", "trellis2/representations"]:
        m = types.ModuleType(p.replace("/", "."))
        m.__path__ = [os.path.join(t2, p)]
        sys.modules[m.__name__] = m
    sys.modules["trellis2.representations"].Mesh = MeshStub
    sys.path.insert(0, t2)
    import trellis2.modules.sparse.conv.conv as spconv
    spconv._backends["flex_gemm"] = GatherConvBackend
    import trellis2.modules.sparse.attention.modules as sam
    from trellis2.modules.sparse import VarLenTensor

    def sp_sdpa(*args):
        f = lambda t: t.feats if isinstance(t, VarLenTensor) else t
        ref = args[0]
        if len(args) == 1:
            q, k, v = f(ref).unbind(1)                     # [N, 3, H, D]
        elif len(args) == 2:
            q, kv = f(args[0]), f(args[1])
            if kv.dim() == 5:
                kv = kv[0]
            k, v = kv.unbind(-3)
        else:
            q, k, v = (f(a) for a in args)
        q, k, v = (t[0] if t.dim() == 4 else t for t in (q, k, v))
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])[0].transpose(0, 1)
        return ref.replace(o) if isinstance(ref, VarLenTensor) else o
    sam.sparse_scaled_dot_product_attention = sp_sdpa


# ---- weights ----------------------------------------------------------------------------------------------------------

REMAP = {"time_in.": "t_embedder.", "guidance_in.": "adaLN_modulation.1.", "img_in.": "input_layer.", "final_layer.": "out_layer.", "blk.": "blocks."}


def gguf_state(path):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(path)
    sd = {}
    for t in r.tensors:
        a = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(tuple(int(x) for x in reversed(t.shape)))
        k = t.name
        for o, n in REMAP.items():
            if k.startswith(o):
                k = n + k[len(o):]
                break
        sd[k] = torch.from_numpy(a.copy())
    return sd


def st_state(path):
    from safetensors.torch import load_file
    return {k: v.float() for k, v in load_file(path).items()}


def flow(models, which):
    import trellis2.models as tm
    base = os.path.join(models, FLOWS[which])
    cfg = json.load(open(base + ".json"))
    args = dict(cfg["args"], dtype="float32")
    m = getattr(tm, cfg["name"])(**args)
    sd = gguf_state(base + "_Q8_0.gguf")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        log(which, "missing", missing[:6], "unexpected", unexpected[:6])
    return m.eval().float()


def decoder(models, which):
    import trellis2.models as tm
    base = os.path.join(models, DECODERS[which])
    cfg = json.load(open(base + ".json"))
    args = dict(cfg["args"], use_fp16=False)
    m = getattr(tm, cfg["name"])(**args)
    missing, unexpected = m.load_state_dict(st_state(base + ".safetensors"), strict=False)
    if missing or unexpected:
        log(which, "decoder missing", missing[:6], "unexpected", unexpected[:6])
    return m.eval().float()


# ---- stages -----------------------------------------------------------------------------------------------------------

def birefnet(models, img, out):
    """TRELLIS.2's rembg.BiRefNet on the CPU: 1024 x 1024 bilinear, ImageNet normalisation, the last prediction's sigmoid as
    an 8-bit mask resized back (PIL's default, bicubic) - and the encoder's features for the check"""
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    m = AutoModelForImageSegmentation.from_pretrained(os.path.join(models, "birefnet"), trust_remote_code=True, dtype=torch.float32).eval().float()
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(img)[None]
    save(out, "rm_in", x[0])
    (x1, x2, x3, x4), _ = m.forward_enc(x)
    save(out, "rm_x1", x1[0])
    save(out, "rm_x4", x4[0])
    preds = m(x)[-1]
    save(out, "rm_logits", preds[0, 0])
    pred = preds.sigmoid()[0].squeeze()
    mask = transforms.ToPILImage()(pred).resize(img.size)
    mask.save(os.path.join(out, "rm_mask.png"))
    img = img.copy()
    img.putalpha(mask)
    return img


def preprocess(models, img, out):
    """Trellis2ImageTo3DPipeline.preprocess_image"""
    has_alpha = img.mode == "RGBA" and not np.all(np.array(img)[:, :, 3] == 255)
    scale = min(1, 1024 / max(img.size))
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    o = img if has_alpha else birefnet(models, img.convert("RGB"), out)
    a = np.array(o)[:, :, 3]
    bb = np.argwhere(a > 0.8 * 255)
    bb = np.min(bb[:, 1]), np.min(bb[:, 0]), np.max(bb[:, 1]), np.max(bb[:, 0])
    center = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    size = int(max(bb[2] - bb[0], bb[3] - bb[1]))
    bb = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    o = np.array(o.crop(bb)).astype(np.float32) / 255
    o = o[:, :, :3] * o[:, :, 3:4]
    o = Image.fromarray((o * 255).astype(np.uint8))
    o.save(os.path.join(out, "pre.png"))
    return o


def dino(models, img, size, out):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from safetensors.torch import load_file
    cfg = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16, patch_size=16,
                          num_register_tokens=4, rope_theta=100.0, layer_norm_eps=1e-5, query_bias=True, key_bias=False,
                          value_bias=True, proj_bias=True, mlp_bias=True, use_gated_mlp=False, hidden_act="gelu",
                          layerscale_value=1.0, image_size=size)
    m = DINOv3ViTModel(cfg)
    sd = load_file(os.path.join(models, "Vision/dinov3-vitl16-pretrain-lvd1689m.safetensors"))
    # the file's names are the original ones; this transformers keeps the layers under `model.`
    layers = m.model.layer if hasattr(m, "model") else m.layer
    pre = "model." if hasattr(m, "model") else ""
    sd = {(pre + k if k.startswith("layer.") else k): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    m.eval()
    # DinoV3FeatureExtractor: LANCZOS to size, /255, ImageNet normalisation, the layers, a parameter-free layer norm
    x = np.array(img.resize((size, size), Image.LANCZOS).convert("RGB")).astype(np.float32) / 255
    x = torch.from_numpy(x).permute(2, 0, 1)[None]
    x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    save(out, f"dino{size}_in", x[0])
    h = m.embeddings(x, bool_masked_pos=None)
    pe = m.rope_embeddings(x)
    for i, layer in enumerate(layers):
        h = layer(h, position_embeddings=pe)
        if i == 0:
            save(out, f"dino{size}_l0", h[0])
    c = F.layer_norm(h, h.shape[-1:])
    save(out, f"cond{size}", c[0])
    return c


class Rec:
    """The flow model as the sampler calls it, its first calls' outputs (and the first call's insides) kept"""
    def __init__(self, model, out, name, keep=2):
        self.m, self.out, self.name, self.keep, self.calls = model, out, name, keep, 0
        self.hooks = []
        for tag, mod in [("in0", model.input_layer), ("blk0", model.blocks[0]), (f"blk{len(model.blocks) - 1}", model.blocks[-1])]:
            self.hooks.append(mod.register_forward_hook(self.hook(tag)))

    def hook(self, tag):
        def f(_m, _i, o):
            if self.calls == 0:
                save(self.out, f"{self.name}_{tag}", o.feats if hasattr(o, "feats") else o[0])
        return f

    def __call__(self, x, t, cond, **kw):
        y = self.m(x, t, cond, **kw)
        if self.calls < self.keep:
            save(self.out, f"{self.name}_call{self.calls}", y.feats if hasattr(y, "feats") else y[0])
            if self.calls == 0:
                save(self.out, f"{self.name}_t0", t)
        self.calls += 1
        return y

    def close(self):
        for h in self.hooks:
            h.remove()


def sample(model, noise, cond, params, out, name, steps, **kw):
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    s = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    rec = Rec(model, out, name)
    p = dict(params, steps=steps)
    t0 = time.time()
    r = s.sample(rec, noise, cond=cond, neg_cond=torch.zeros_like(cond), verbose=False, **p, **kw)
    rec.close()
    for i, x in enumerate(r.pred_x_t):
        save(out, f"{name}_x{i + 1}", x.feats if hasattr(x, "feats") else x[0])
    log(f"{name}: {rec.calls} calls, {time.time() - t0:.0f} s")
    return r.samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picture")
    ap.add_argument("out")
    ap.add_argument("--t2", default="/var/mnt/2TBSSD/nextsycl-ref/TRELLIS.2")
    ap.add_argument("--models", default=os.path.expanduser("~/models/trellis2"))
    ap.add_argument("--pipeline", default="512", choices=["512", "1024_cascade"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=49152)
    ap.add_argument("--stages", default="pre,dino,ss,shape,tex,sdec,tdec")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    setup(a.t2)
    from trellis2.modules.sparse import SparseTensor
    stages = a.stages.split(",")
    pj = json.load(open(os.path.join(a.models, "pipeline.json")))["args"]
    infop = os.path.join(a.out, "info.json")
    info = json.load(open(infop)) if os.path.exists(infop) else {}
    info.update(pipeline=a.pipeline, seed=a.seed, steps=a.steps)
    cascade = a.pipeline == "1024_cascade"

    if "pre" in stages:
        img = preprocess(a.models, Image.open(a.picture), a.out)
    else:
        img = Image.open(os.path.join(a.out, "pre.png"))
    if "dino" in stages:
        dino(a.models, img, 512, a.out)
        if cascade:
            dino(a.models, img, 1024, a.out)
        log("dino done")
    if not set(stages) & {"ss", "shape", "tex", "sdec", "tdec"}:
        json.dump(info, open(infop, "w"), indent=1)
        return
    cond512 = load(a.out, "cond512")[None]
    cond1024 = load(a.out, "cond1024")[None] if cascade else None

    # the noise in the pipeline's order (seed, then the structure's, the shape's, the cascade's, the texture's)
    g = torch.manual_seed(a.seed)
    ss_noise = torch.randn(1, 8, 16, 16, 16)
    if "ss" in stages:
        save(a.out, "ss_noise", ss_noise[0])
        m = flow(a.models, "ss")
        z = sample(m, ss_noise, cond512, pj["sparse_structure_sampler"]["params"], a.out, "ss", a.steps)
        del m
        save(a.out, "ss_z", z[0])
        d = decoder(a.models, "ss")
        logits = d(z)
        save(a.out, "ss_logits", logits[0, 0])
        dec = logits > 0
        dec = F.max_pool3d(dec.float(), 2, 2, 0) > 0.5           # ss_res 32 for both pipelines here
        coords = torch.argwhere(dec)[:, [0, 2, 3, 4]].int()
        save(a.out, "ss_coords", coords[:, 1:])
        log("ss: tokens", coords.shape[0])
        info["ss_tokens"] = int(coords.shape[0])
    if not set(stages) & {"shape", "tex", "sdec", "tdec"}:
        json.dump(info, open(infop, "w"), indent=1)
        return
    c3 = load(a.out, "ss_coords").int()
    coords = torch.cat([torch.zeros(c3.shape[0], 1, dtype=torch.int32), c3], 1)
    shape_noise = torch.randn(coords.shape[0], 32)
    mean = torch.tensor(pj["shape_slat_normalization"]["mean"])[None]
    std = torch.tensor(pj["shape_slat_normalization"]["std"])[None]
    res = 512
    if "shape" in stages:
        save(a.out, "shape_noise", shape_noise)
        m = flow(a.models, "shape512")
        slat = sample(m, SparseTensor(feats=shape_noise, coords=coords), cond512, pj["shape_slat_sampler"]["params"], a.out, "shape", a.steps)
        del m
        slat = slat * std + mean
        save(a.out, "shape_slat", slat.feats)
        if cascade:
            d = decoder(a.models, "shape")
            hr = d.upsample(slat, upsample_times=4)
            del d
            save(a.out, "hr_up", hr[:, 1:])
            hr_res = 1024
            while True:
                q = torch.cat([hr[:, :1], ((hr[:, 1:] + 0.5) / 512 * (hr_res // 16)).int()], 1)
                c2 = q.unique(dim=0)
                if c2.shape[0] < a.max_tokens or hr_res == 1024:
                    break
                hr_res -= 128
            save(a.out, "hr_coords", c2[:, 1:])
            info["res"] = hr_res
            info["hr_tokens"] = int(c2.shape[0])
            log("cascade: tokens", c2.shape[0], "res", hr_res)
            hr_noise = torch.randn(c2.shape[0], 32)
            save(a.out, "shape_hr_noise", hr_noise)
            m = flow(a.models, "shape1024")
            slat = sample(m, SparseTensor(feats=hr_noise, coords=c2.int()), cond1024, pj["shape_slat_sampler"]["params"], a.out, "shape_hr", a.steps)
            del m
            slat = slat * std + mean
            save(a.out, "shape_hr_slat", slat.feats)
        else:
            info["res"] = 512
    res = info.get("res", 512)
    if cascade:
        c3 = load(a.out, "hr_coords").int()
        coords = torch.cat([torch.zeros(c3.shape[0], 1, dtype=torch.int32), c3], 1)
        if "shape" not in stages:
            torch.randn(coords.shape[0], 32)                # the cascade's noise, drawn before the texture's
    shape_slat = SparseTensor(feats=load(a.out, "shape_hr_slat" if cascade else "shape_slat"), coords=coords)
    tex_noise = torch.randn(coords.shape[0], 32)
    if "tex" in stages:
        save(a.out, "tex_noise", tex_noise)
        m = flow(a.models, "tex1024" if cascade else "tex512")
        norm = (shape_slat - mean) / std
        tslat = sample(m, norm.replace(feats=tex_noise), cond1024 if cascade else cond512, pj["tex_slat_sampler"]["params"], a.out, "tex",
                       a.steps, concat_cond=norm)
        del m
        tm_ = torch.tensor(pj["tex_slat_normalization"]["mean"])[None]
        ts_ = torch.tensor(pj["tex_slat_normalization"]["std"])[None]
        tslat = tslat * ts_ + tm_
        save(a.out, "tex_slat", tslat.feats)
    if "sdec" in stages or "tdec" in stages:
        d = decoder(a.models, "shape")
        d.set_resolution(res)
        hooks = []
        ups = [blocks[-1] for blocks in d.blocks[:-1]]
        for i, up in enumerate(ups):
            def f(_m, inp, o, i=i):
                save(a.out, f"sdec_c{i}", inp[0].coords[:, 1:])
                save(a.out, f"sdec_sub{i}", o[1].feats)
                save(a.out, f"sdec_up{i}", o[0].feats)
            hooks.append(up.register_forward_hook(f))

        def fo(_m, inp, o):
            save(a.out, "sdec_h", o.feats)
            save(a.out, "sdec_coords", o.coords[:, 1:])
        hooks.append(d.output_layer.register_forward_hook(fo))
        meshes, subs = d(shape_slat, return_subs=True)
        for h in hooks:
            h.remove()
        save(a.out, "mesh_v", meshes[0].vertices)
        save(a.out, "mesh_f", meshes[0].faces)
        info["mesh"] = [int(meshes[0].vertices.shape[0]), int(meshes[0].faces.shape[0])]
        log("sdec: mesh", info["mesh"])
        del d
        if "tdec" in stages:
            td = decoder(a.models, "tex")
            tslat = SparseTensor(feats=load(a.out, "tex_slat"), coords=coords)
            v = td(tslat, guide_subs=subs) * 0.5 + 0.5
            save(a.out, "tdec", v.feats)
            save(a.out, "tdec_coords", v.coords[:, 1:])
            log("tdec:", tuple(v.feats.shape))
    json.dump(info, open(infop, "w"), indent=1)
    log("done", info)


if __name__ == "__main__":
    main()
