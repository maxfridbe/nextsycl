"""reference/pixal3d/ref.py: Pixal3D's conditioning and flows on the CPU through its own modules (TencentARC/Pixal3D at
f7cf384: `git clone https://github.com/TencentARC/Pixal3D && git -C Pixal3D checkout f7cf38429b0bd264f1995f0f8743a88b1c728b94`),
on the files nextsycl reads (Aero-Ex/Pixal3D-GGUF's Q8_0 flows dequantized exactly, TRELLIS.2's DINOv3), dumping
what `nextsycl model3d check DIR --model pixal3d --stages pixal` compares. The CUDA stand-ins are
reference/trellis2/ref.py's.

    python ref.py PICTURE OUT --coords512 helmet512/ss_coords.npy --coords1024 helmet1024/hr_coords.npy
                  [--px REPO] [--models ~/models/pixal3d] [--t2models ~/models/trellis2] [--fov RADIANS]

  pre.png                     Pixal3DImageTo3DPipeline.preprocess_image (the box 1.1 times, black behind)
  px_<stage>_global.npy       DinoV3ProjFeatureExtractor's global tokens (CLS + 4 registers) [5, 1024]
  px_<stage>_proj.npy         its projected features at the stage's voxels (ProjGrid: the official projection);
                              stages ss (512 px, 16^3, 1024 wide), shape512 (512 px, 32^3), shape1024 and tex (1024 px,
                              64^3), the last three [low-res | low-res again] (NAF's half without NAF: the port's fallback)
  px_<stage>_noise / _t0 / _in0 / _blk0 / _blk29 / _call0 / _call1.npy   each flow's first calls (cond, then zeros)
  px_info.json                the camera (fov, distance)

The stages' voxels are given (any set serves a forward pass): the structure's full 16^3 grid, --coords512 for the 512
shape, --coords1024 for the 1024 shape and texture (whose concatenated shape latent is zeros + noise here).
"""
import argparse
import json
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "trellis2"))
import ref as t2ref                                             # noqa: E402  (the stand-ins and loaders)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
from PIL import Image                                           # noqa: E402

STAGES = {
    "ss": dict(file="Sparse/ss_flow_img_dit_1_3B_64_bf16", size=512, grid=16, naf=False),
    "shape512": dict(file="shape/slat_flow_img2shape_dit_1_3B_512_bf16", size=512, grid=32, naf=True),
    "shape1024": dict(file="shape/slat_flow_img2shape_dit_1_3B_1024_bf16", size=1024, grid=64, naf=True),
    "tex": dict(file="texture/slat_flow_imgshape2tex_dit_1_3B_1024_bf16", size=1024, grid=64, naf=True),
}


def setup(px):
    def stub(name, **kw):
        m = types.ModuleType(name)
        m.__dict__.update(kw)
        sys.modules[name] = m
        return m
    for n in ["cumesh", "cumesh.remeshing", "nvdiffrast", "nvdiffrast.torch", "flex_gemm", "flex_gemm.ops", "o_voxel", "o_voxel.postprocess"]:
        stub(n)
    stub("flex_gemm.ops.grid_sample", grid_sample_3d=None)
    stub("flex_gemm.ops.spconv", sparse_submanifold_conv3d=None, set_algorithm=None, set_hashmap_ratio=None)
    stub("o_voxel.convert", flexible_dual_grid_to_mesh=t2ref.fdg_to_mesh)
    for p in ["pixal3d", "pixal3d/pipelines", "pixal3d/representations", "pixal3d/trainers", "pixal3d/trainers/flow_matching",
              "pixal3d/trainers/flow_matching/mixins", "pixal3d/utils"]:
        m = types.ModuleType(p.replace("/", "."))
        m.__path__ = [os.path.join(px, p)]
        sys.modules[m.__name__] = m
    sys.modules["pixal3d.representations"].Mesh = t2ref.MeshStub
    stub("pixal3d.utils.dist_utils", read_file_dist=None)
    sys.path.insert(0, px)
    import pixal3d.modules.sparse.conv.conv as spconv
    spconv._backends["flex_gemm"] = t2ref.GatherConvBackend
    import pixal3d.modules.sparse.attention.modules as sam
    from pixal3d.modules.sparse import VarLenTensor

    def sp_sdpa(*args):
        f = lambda t: t.feats if isinstance(t, VarLenTensor) else t
        ref = args[0]
        if len(args) == 1:
            q, k, v = f(ref).unbind(1)
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


def preprocess(img):
    """Pixal3DImageTo3DPipeline.preprocess_image for a picture with alpha (bg black)"""
    scale = min(1, 1024 / max(img.size))
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    a = np.array(img)[:, :, 3]
    bb = np.argwhere(a > 0.8 * 255)
    bb = np.min(bb[:, 1]), np.min(bb[:, 0]), np.max(bb[:, 1]), np.max(bb[:, 0])
    center = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    size = int(max(bb[2] - bb[0], bb[3] - bb[1]) * 1.1)
    bb = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    o = np.array(img.crop(bb)).astype(np.float32) / 255
    rgb, al = o[:, :, :3], o[:, :, 3:4]
    o = rgb * al + np.zeros(3, np.float32) * (1.0 - al)
    return Image.fromarray((np.clip(o, 0, 1) * 255).astype(np.uint8))


def dino_tokens(t2models, img, size):
    """DINOv3 as DinoV3ProjFeatureExtractor.extract_features runs it: [1, 5 + p, 1024]"""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from safetensors.torch import load_file
    cfg = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16, patch_size=16,
                          num_register_tokens=4, rope_theta=100.0, layer_norm_eps=1e-5, query_bias=True, key_bias=False,
                          value_bias=True, proj_bias=True, mlp_bias=True, use_gated_mlp=False, hidden_act="gelu",
                          layerscale_value=1.0, image_size=size)
    m = DINOv3ViTModel(cfg)
    sd = load_file(os.path.join(t2models, "Vision/dinov3-vitl16-pretrain-lvd1689m.safetensors"))
    pre = "model." if hasattr(m, "model") else ""
    m.load_state_dict({(pre + k if k.startswith("layer.") else k): v for k, v in sd.items()}, strict=True)
    m.eval()
    layers = m.model.layer if hasattr(m, "model") else m.layer
    x = np.array(img.resize((size, size), Image.LANCZOS).convert("RGB")).astype(np.float32) / 255
    x = torch.from_numpy(x).permute(2, 0, 1)[None]
    x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    h = m.embeddings(x, bool_masked_pos=None)
    pe = m.rope_embeddings(x)
    for layer in layers:
        h = layer(h, position_embeddings=pe)
    return F.layer_norm(h, h.shape[-1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picture")
    ap.add_argument("out")
    ap.add_argument("--px", default="/var/mnt/2TBSSD/nextsycl-ref/Pixal3D")
    ap.add_argument("--models", default=os.path.expanduser("~/models/pixal3d"))
    ap.add_argument("--t2models", default=os.path.expanduser("~/models/trellis2"))
    ap.add_argument("--coords512", required=True)
    ap.add_argument("--coords1024", required=True)
    ap.add_argument("--fov", type=float, default=0.8575560450553894)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    setup(a.px)
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid
    import pixal3d.models as pm
    from pixal3d.modules.sparse import SparseTensor
    # the camera: inference.py's distance_from_fov with the manual fov, mesh_scale 1, 512 px
    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = (torch.tensor([-1.0, 0.0, 0.0]) @ R.T) / 1.0 / 2
    f_px = float((16.0 / torch.tan(torch.tensor(a.fov / 2.0))) * 512 / 32.0)
    distance = f_px * gp[0].item() / (0 - 256.0) - gp[1].item()
    json.dump({"fov": a.fov, "distance": distance}, open(os.path.join(a.out, "px_info.json"), "w"))
    img = preprocess(Image.open(a.picture))
    img.save(os.path.join(a.out, "pre.png"))
    tokens = {s: dino_tokens(a.t2models, img, s) for s in (512, 1024)}
    c512 = torch.from_numpy(np.load(a.coords512)).int()
    c1024 = torch.from_numpy(np.load(a.coords1024)).int()
    grid16 = torch.stack(torch.meshgrid(*[torch.arange(16)] * 3, indexing="ij"), -1).reshape(-1, 3).int()
    torch.manual_seed(a.seed)
    for name, st in STAGES.items():
        z = tokens[st["size"]]
        np_ = st["size"] // 16
        patches = z[:, 5:].reshape(1, np_, np_, -1)
        pg = ProjGrid(grid_resolution=st["grid"], image_resolution=st["size"])
        lr = pg(patches, torch.tensor([a.fov]), torch.tensor([distance]), torch.tensor([1.0]))      # [1, R^3, 1024]
        proj = torch.cat([lr, lr], -1) if st["naf"] else lr
        coords = grid16 if name == "ss" else (c512 if name == "shape512" else c1024)
        g = st["grid"]
        idx = (coords[:, 0].long() * g + coords[:, 1].long()) * g + coords[:, 2].long()
        zp = proj[0, idx]
        glob = z[:, :5]
        t2ref.save(a.out, f"px_{name}_global", glob[0])
        t2ref.save(a.out, f"px_{name}_proj", zp)
        base = os.path.join(a.models, st["file"])
        cfg = json.load(open(base + ".json"))
        m = getattr(pm, cfg["name"])(**dict(cfg["args"], dtype="float32"))
        # some small F16 tensors are stored reshaped (the RMS gammas as [6, 256]): the same values in order
        own = m.state_dict()
        sd = {k: (v.reshape(own[k].shape) if k in own and v.shape != own[k].shape and v.numel() == own[k].numel() else v)
              for k, v in t2ref.gguf_state(base + "_Q8_0.gguf").items()}
        # the structure file carries the model's rotary buffer as half reals: the model's own (complex) phases stay
        sd.pop("rope_phases", None)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        t2ref.log(name, "missing", [k for k in missing if "rope" not in k][:5], "unexpected", unexpected[:5])
        m = m.eval().float()
        t = torch.tensor([1000.0])
        if name == "ss":
            x = torch.randn(1, 8, 16, 16, 16)
            t2ref.save(a.out, f"px_{name}_noise", x[0])
            cond = {"global": glob, "proj": zp[None]}
            neg = {"global": torch.zeros_like(glob), "proj": torch.zeros_like(zp)[None]}
            kw = {}
        else:
            n = coords.shape[0]
            full = torch.cat([torch.zeros(n, 1, dtype=torch.int32), coords], 1)
            x = SparseTensor(feats=torch.randn(n, 32), coords=full)
            t2ref.save(a.out, f"px_{name}_noise", x.feats)
            cond = {"global": glob, "proj": SparseTensor(feats=zp, coords=full)}
            neg = {"global": torch.zeros_like(glob), "proj": SparseTensor(feats=torch.zeros_like(zp), coords=full)}
            kw = {}
            if name == "tex":
                cc = torch.randn(n, 32)
                t2ref.save(a.out, f"px_{name}_concat", cc)
                kw = {"concat_cond": SparseTensor(feats=cc, coords=full)}
        rec = t2ref.Rec(m, a.out, f"px_{name}")
        rec(x, t, cond, **kw)
        rec(x, t, neg, **kw)
        rec.close()
        t2ref.log(name, "done")
    t2ref.log("done")


if __name__ == "__main__":
    main()
