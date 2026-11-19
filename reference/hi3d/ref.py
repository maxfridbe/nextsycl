"""reference/hi3d/ref.py: Hi3D v1.0 (hi3d-official/Hi3Dv1.0, the `sparc3d` package) on the CPU in float32 through its
own modules, on the SAME files nextsycl reads, dumping every stage the engine is checked against
(`nextsycl model3d check OUT --model hi3d-1.0 --stages ...`).

    git clone https://github.com/hi3d-official/Hi3Dv1.0 SRC     (checked at 13657ad)
    python -I ref.py --src SRC --models DIR --image PICTURE --mesh COARSE.glb --out OUT
                     [--stages pre,cond,vox,dit,steps,vae,post] [--structure 64|128] [--seed 0] [--steps 25]
                     [--updates K] [--latents FILE.npy] [--threads 16]

DIR is the model directory as the card lays it out: model_index.json, scheduler/, vae/, transformer_512/,
transformer_1024/, dinov2-large/ and clip-vit-large-patch14/ (each with its config.json and model.safetensors).

Upstream is CUDA-only. Stand-ins for the CUDA parts (the modules themselves are untouched):
  - flash_attn's varlen functions: torch SDPA in float32 per sequence (exact at batch 1; the decoder's window
    attention keeps upstream's `qkv.half()` rounding of its inputs);
  - spconv's SubMConv3d: a gather - a [N, 27] neighbour map from sorted coordinate keys (tap (a, b, c) of the weight
    [Co, 3, 3, 3, Ci] reads the voxel at (x + a - 1, y + b - 1, z + c - 1); missing neighbours zero), 27 products
    accumulated in float32;
  - the Triton surface voxelizer: the same separating-axis test in float32 numpy over each triangle's box.
The transformers are run in float32 (their fp16 weights widened exactly; upstream runs their torso in fp16 on CUDA).

Stages (each reads what it needs from OUT, so they run one at a time):
  pre:   prep.png (the framed square), dino_in.npy [3,756,756], clip_in.npy [3,224,224] (normalized)
  cond:  dino_raw.npy [2917,1024] (last hidden state), clip_raw.npy [1024] (pooler), cond.npy [2917,2048]
  vox:   coords_S.npy [N,3] int32 (the voxels of --mesh at structure S = 64, or 128 from mesh_64 when absent), vox_S.json
  dit:   noise_S.npy [N,32] (torch.randn, the CPU generator at --seed), sigmas_S.npy, dit_S_<tap>.npy for the first
         call at sigma_hat[0] (input, in0, in1, blk0, blk23, out0, out1, vel), dit_S_neg.npy (the zero condition)
  steps: x_S_<k>.npy after each update, latents_S.npy (denormalized) [N,32]
  vae:   from latents_S.npy (or --latents): vae_S_z2f.npy, vae_S_mid.npy, vae_S_up<i>_{coords,logits}.npy (each
         unpooling's parent coordinates and its 8 occupancy logits), vae_S_coords.npy [M,3], vae_S_sign.npy [M,8],
         vae_S_feat.npy [M,32] (activated); mesh_S_raw_{v,f}.npy (FlexiCubes), mesh_S_{v,f}.npy (holes filled)
  post:  final_S_{v,f}.npy and final_S.glb from mesh_S (components, pyfqmr to 1M faces, recentred, unit radius)
"""
import argparse
import itertools
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--image")
ap.add_argument("--mesh")
ap.add_argument("--out", required=True)
ap.add_argument("--stages", default="pre,cond,vox,dit")
ap.add_argument("--structure", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--updates", type=int, default=0, help="stop the sampler after this many updates (0: all)")
ap.add_argument("--latents")
ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.src)
S = a.structure
O = lambda n: os.path.join(a.out, n)


def save(name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        t = t.float().numpy() if t.is_floating_point() else t.numpy()
    np.save(O(name), t)


# ---- stand-ins ------------------------------------------------------------------------------------------------------
fa = types.ModuleType("flash_attn")


def _sdpa(q, k, v, scale):
    # q, k, v [L, H, D] float32 -> [L, H, D]; query blocks keep the score table small
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))
    out = torch.empty_like(q)
    for i in range(0, q.shape[1], 2048):
        out[:, i:i + 2048] = F.scaled_dot_product_attention(q[:, i:i + 2048], k, v, scale=scale)
    return out.transpose(0, 1)


def varlen_qkvpacked(qkv, cu, max_seqlen, dropout_p=0.0, softmax_scale=None, **kw):
    cu = [int(c) for c in cu]
    lens = [b - c for c, b in zip(cu[:-1], cu[1:])]
    if len(set(lens)) == 1 and len(lens) > 1:
        # equal windows (the decoder's padded patches): one batched call
        L = lens[0]
        x = qkv.float().reshape(-1, L, 3, qkv.shape[2], qkv.shape[3])
        q, k, v = (x[:, :, i].transpose(1, 2) for i in range(3))
        o = F.scaled_dot_product_attention(q, k, v, scale=softmax_scale).transpose(1, 2)
        return o.reshape(-1, qkv.shape[2], qkv.shape[3]).to(qkv.dtype)
    out = torch.empty(qkv.shape[0], qkv.shape[2], qkv.shape[3], dtype=qkv.dtype)
    for c, b in zip(cu[:-1], cu[1:]):
        q, k, v = qkv[c:b].float().unbind(1)
        out[c:b] = _sdpa(q, k, v, softmax_scale).to(qkv.dtype)
    return out


def varlen_kvpacked(q, kv, cu_q, cu_k, mq, mk, dropout_p=0.0, softmax_scale=None, **kw):
    cu_q = [int(c) for c in cu_q]
    cu_k = [int(c) for c in cu_k]
    out = torch.empty_like(q)
    for (q0, q1), (k0, k1) in zip(zip(cu_q[:-1], cu_q[1:]), zip(cu_k[:-1], cu_k[1:])):
        k, v = kv[k0:k1].float().unbind(1)
        out[q0:q1] = _sdpa(q[q0:q1].float(), k, v, softmax_scale).to(q.dtype)
    return out


fa.flash_attn_varlen_qkvpacked_func = varlen_qkvpacked
fa.flash_attn_varlen_kvpacked_func = varlen_kvpacked
# (handed to sparc3d's two lazy imports only: transformers probes sys.modules for a real flash_attn)
import sparc3d.models.autoencoders.blocks as _blocks  # noqa: E402
import sparc3d.nn.sparse.attention.full_attn as _full  # noqa: E402

_blocks._flash_attn = lambda: fa
_full._flash_attn = lambda: fa

from sparc3d.ops import sparse_backend as sb  # noqa: E402


class FakeSparse:
    """spconv's SparseConvTensor as far as sparc3d uses it: features, indices, and the neighbour map it caches"""

    def __init__(self, features, indices, nmap=None):
        self.features, self.indices, self.nmap = features, indices, nmap

    def replace_feature(self, f):
        return FakeSparse(f, self.indices, self.nmap)


def nmap_of(coords):
    """[N, 27] rows of each voxel's neighbours (tap (a, b, c) at offset (a-1, b-1, c-1)), -1 where absent"""
    c = coords[:, 1:].long() + 1
    m = int(c.max()) + 2
    key = (c[:, 0] * m + c[:, 1]) * m + c[:, 2]
    sk, order = torch.sort(key)
    out = torch.full((c.shape[0], 27), -1, dtype=torch.long)
    for t, (da, db, dc) in enumerate(itertools.product((-1, 0, 1), repeat=3)):
        q = ((c[:, 0] + da) * m + (c[:, 1] + db)) * m + (c[:, 2] + dc)
        pos = torch.searchsorted(sk, q).clamp(max=sk.shape[0] - 1)
        hit = sk[pos] == q
        out[hit, t] = order[pos[hit]]
    return out


class GatherConv(torch.nn.Module):
    """A submanifold 3x3x3 convolution as a gather and 27 float32 products (the weight as spconv stores it)"""

    def __init__(self, ci, co, kernel_size=3, stride=1, padding=None, bias=True):
        super().__init__()
        assert kernel_size == 3
        self.weight = torch.nn.Parameter(torch.empty(co, 3, 3, 3, ci))
        self.bias = torch.nn.Parameter(torch.empty(co)) if bias else None
        self.in_channels, self.out_channels = ci, co
        setattr(self, sb._SPARSE_CONV_FLAG, True)

    def forward(self, x):
        if x.nmap is None:
            x.nmap = nmap_of(x.indices)
        dt = x.features.dtype
        f = x.features.to(self.weight.dtype)
        w = self.weight.reshape(self.out_channels, 27, self.in_channels)
        n = f.shape[0]
        out = torch.zeros(n, self.out_channels, dtype=f.dtype)
        fz = torch.cat([f, torch.zeros(1, f.shape[1], dtype=f.dtype)])
        for r in range(0, n, 262144):
            idx = x.nmap[r:r + 262144]
            idx = torch.where(idx < 0, n, idx)
            acc = torch.zeros(idx.shape[0], self.out_channels, dtype=f.dtype)
            for t in range(27):
                acc += fz[idx[:, t]] @ w[:, t, :].T
            out[r:r + 262144] = acc
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return FakeSparse(out.to(dt), x.indices, x.nmap)


def fake_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=True):
    return GatherConv(in_channels, out_channels, kernel_size, stride, padding, bias)


sb.Conv3d = fake_conv3d
sb.sparse_tensor = lambda feats, coords, spatial_shape=None: FakeSparse(feats, coords.int().contiguous())
sb.get_feats = lambda t: t.features
sb.replace_feats = lambda t, f: t.replace_feature(f)


def adaptive_vox_cpu(mesh, resolution=64, target_num=80000, device="cpu"):
    """sparc3d.ops.voxelize.adaptive_vox with the Triton kernels as float32 numpy: the same normalisation, triangle
    boxes and separating-axis test; coordinates sorted by (z << 22 | y << 11 | x) as the kernel's unique keys"""
    v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)
    scale = 1.0
    while True:
        res = int(resolution * scale)
        vmin, vmax = v.min(0)[0], v.max(0)[0]
        emin, emax = vmin - 1e-6, vmax + 1e-6
        center = (emin + emax) / 2.0
        s = (res - 3 * 2 - 2 * 1e-6) / (emax - emin).max()
        nv = ((v - center) * s + (res / 2.0)).numpy().astype(np.float32)
        coords = surface_voxels(nv, f, res)
        if coords.shape[0] <= target_num:
            break
        scale *= 0.95
    return torch.from_numpy(coords).int(), res


def surface_voxels(nv, f, res):
    tri = nv[f]  # [F, 3 verts, 3]
    lo = np.maximum(np.floor(tri.min(1)).astype(np.int32), 0)
    hi = np.minimum(np.floor(tri.max(1)).astype(np.int32), res - 1)
    ext = np.maximum(hi - lo + 1, 0)
    keys = []
    f32 = np.float32
    for e in sorted(set(map(tuple, ext.tolist()))):
        sel = np.nonzero((ext == np.array(e)).all(1))[0]
        if len(sel) == 0 or min(e) <= 0:
            continue
        for ox, oy, oz in itertools.product(range(e[0]), range(e[1]), range(e[2])):
            x, y, z = lo[sel, 0] + ox, lo[sel, 1] + oy, lo[sel, 2] + oz
            c = np.stack([x, y, z], 1).astype(f32) + f32(0.5)
            t = tri[sel] - c[:, None, :]
            hit = overlaps(t)
            keys.append(((z[hit].astype(np.int64) << 22) | (y[hit].astype(np.int64) << 11) | x[hit].astype(np.int64)))
    k = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    m = (1 << 11) - 1
    return np.stack([k & m, (k >> 11) & m, (k >> 22) & m], 1).astype(np.int32)


def overlaps(t):
    f32 = np.float32
    v0, v1, v2 = t[:, 0], t[:, 1], t[:, 2]
    ok = np.ones(t.shape[0], bool)
    for ax in range(3):
        ok &= (np.minimum(np.minimum(v0[:, ax], v1[:, ax]), v2[:, ax]) <= f32(0.5)) & (np.maximum(np.maximum(v0[:, ax], v1[:, ax]), v2[:, ax]) >= f32(-0.5))
    e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
    zero = np.zeros(t.shape[0], f32)
    for e in (e0, e1, e2):
        for axis in ((zero, e[:, 2], -e[:, 1]), (-e[:, 2], zero, e[:, 0]), (e[:, 1], -e[:, 0], zero)):
            ax_, ay_, az_ = axis
            p0 = v0[:, 0] * ax_ + v0[:, 1] * ay_ + v0[:, 2] * az_
            p1 = v1[:, 0] * ax_ + v1[:, 1] * ay_ + v1[:, 2] * az_
            p2 = v2[:, 0] * ax_ + v2[:, 1] * ay_ + v2[:, 2] * az_
            r = f32(0.5) * (np.abs(ax_) + np.abs(ay_) + np.abs(az_))
            ok &= (np.minimum(np.minimum(p0, p1), p2) <= r) & (np.maximum(np.maximum(p0, p1), p2) >= -r)
    n = np.cross(e0, e1).astype(f32)
    off = -(n[:, 0] * v0[:, 0] + n[:, 1] * v0[:, 1] + n[:, 2] * v0[:, 2])
    rad = f32(0.5) * (np.abs(n[:, 0]) + np.abs(n[:, 1]) + np.abs(n[:, 2]))
    return ok & (np.abs(off) <= rad)


import sparc3d.ops  # noqa: E402

from sparc3d.pipelines import loader  # noqa: E402
from sparc3d.pipelines.sparc3d_pipeline import PASS_BUDGETS, STRUCTURE_RESOLUTIONS, Sparc3DPipeline  # noqa: E402

stages = a.stages.split(",")
from pathlib import Path  # noqa: E402

root = Path(os.path.abspath(a.models))
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f} s] {m}", flush=True)


# ---- pre / cond ------------------------------------------------------------------------------------------------------
if "pre" in stages or "cond" in stages:
    from sparc3d.models.conditioners.image import ImageConditioner
    from sparc3d.utils.image import prepare_image

    prepared = prepare_image(Image.open(a.image))
    prepared.save(O("prep.png"))
    cond = ImageConditioner(dino_model_id=str(root / "dinov2-large"), clip_model_id=str(root / "clip-vit-large-patch14"),
                            local_files_only=True)
    base = cond._pil_tensor(prepared, torch.device("cpu"))
    dino_in = (F.interpolate(base, size=(756, 756), mode="bilinear", align_corners=False) - cond._dino_mean) / cond._dino_std
    clip_in = (cond._image_tensor(prepared, 224, Image.Resampling.LANCZOS, torch.device("cpu")) - cond._clip_mean) / cond._clip_std
    save("dino_in.npy", dino_in[0])
    save("clip_in.npy", clip_in[0])
    log("pre")
    if "cond" in stages:
        d = cond.dino(pixel_values=dino_in).last_hidden_state[0]
        c = cond.clip(pixel_values=clip_in).pooler_output[0]
        save("dino_raw.npy", d)
        save("clip_raw.npy", c)
        full = cond(prepared)[0]
        save("cond.npy", full)
        log(f"cond {tuple(full.shape)}")
    del cond

# ---- vox ---------------------------------------------------------------------------------------------------------------
if "vox" in stages:
    import trimesh

    from sparc3d.sparcubes.postprocess import validate_mesh

    if a.mesh:
        mesh = validate_mesh(trimesh.load(a.mesh, force="mesh", process=False))
    else:
        prev = STRUCTURE_RESOLUTIONS[STRUCTURE_RESOLUTIONS.index(S) - 1]
        mesh = trimesh.Trimesh(np.load(O(f"mesh_{prev}_v.npy")), np.load(O(f"mesh_{prev}_f.npy")), process=False)
    coords, res = adaptive_vox_cpu(mesh, S, PASS_BUDGETS[STRUCTURE_RESOLUTIONS.index(S)])
    save(f"coords_{S}.npy", coords)
    json.dump({"structure": S, "grid": res, "voxels": int(coords.shape[0])}, open(O(f"vox_{S}.json"), "w"))
    log(f"vox {S}: {coords.shape[0]} voxels at {res}")


def transformer():
    m = loader.load_transformer(root, "transformer_512" if S == 64 else "transformer_1024").float()
    m.dtype = torch.float32
    return m


def sample_input():
    from sparc3d.nn.sparse.basic import SparseTensor

    coords = torch.from_numpy(np.load(O(f"coords_{S}.npy"))).int()
    g = torch.Generator(device="cpu")
    g.manual_seed(a.seed)
    noise = torch.randn((coords.shape[0], 32), generator=g, device="cpu", dtype=torch.float32)
    bc = torch.cat((torch.zeros((coords.shape[0], 1), dtype=torch.int32), coords), 1)
    return SparseTensor(feats=noise, coords=bc), noise


# ---- dit -------------------------------------------------------------------------------------------------------------
if "dit" in stages:
    tr = transformer()
    sched = loader.load_scheduler(root)
    x, noise = sample_input()
    save(f"noise_{S}.npy", noise)
    ts = sched.set_timesteps(a.steps, num_tokens=x.feats.shape[0])
    save(f"sigmas_{S}.npy", sched.sigmas)
    save(f"sigma_hats_{S}.npy", ts)
    cond = torch.from_numpy(np.load(O("cond.npy")))[None]
    taps = {}

    def hook(name):
        def f(mod, inp, out):
            taps[name] = out.feats.clone()
        return f

    hs = [tr.input_layer.register_forward_hook(hook("input")), tr.input_blocks[0].register_forward_hook(hook("in0")),
          tr.input_blocks[1].register_forward_hook(hook("in1")), tr.blocks[0].register_forward_hook(hook("blk0")),
          tr.blocks[-1].register_forward_hook(hook("blk23")), tr.out_blocks[0].register_forward_hook(hook("out0")),
          tr.out_blocks[1].register_forward_hook(hook("out1"))]
    t = (1000.0 * ts[0]).reshape(1)
    v = tr(x, t, cond).feats
    for k, val in taps.items():
        save(f"dit_{S}_{k}.npy", val)
    save(f"dit_{S}_vel.npy", v)
    for h in hs:
        h.remove()
    log(f"dit {S}: first call at sigma {float(ts[0]):.5f}, {x.feats.shape[0]} tokens")
    vn = tr(x, t, torch.zeros_like(cond)).feats
    save(f"dit_{S}_neg.npy", vn)
    log("dit: the zero condition")
    del tr

# ---- steps -------------------------------------------------------------------------------------------------------------
if "steps" in stages:
    tr = transformer()
    sched = loader.load_scheduler(root)
    x, noise = sample_input()
    cond = torch.from_numpy(np.load(O("cond.npy")))[None]
    neg = torch.zeros_like(cond)
    ts = sched.set_timesteps(a.steps, num_tokens=x.feats.shape[0])
    for i, sh in enumerate(ts):
        v = Sparc3DPipeline._guided_velocity(tr, x, sh, cond, neg)
        pred, ns = sched.predict(v, sh, x.feats)
        v2 = Sparc3DPipeline._guided_velocity(tr, x.replace(pred), ns, cond, neg)
        x = x.replace(sched.step(v, sh, x.feats, second_output=v2))
        save(f"x_{S}_{i}.npy", x.feats)
        log(f"update {i + 1}/{len(ts)} (sigma {float(sh):.4f})")
        if a.updates and i + 1 >= a.updates:
            break
    else:
        mean, std = Sparc3DPipeline._latent_stats(tr, x)
        save(f"latents_{S}.npy", x.feats * std + mean)
    del tr

# ---- vae ---------------------------------------------------------------------------------------------------------------
if "vae" in stages:
    from sparc3d.models.autoencoders import ptv3
    from sparc3d.sparcubes.extract import point_to_mesh
    from sparc3d.sparcubes.fill_hole import fill_mesh_holes

    vae = loader.load_vae(root)
    dec = vae.decoder
    lat = torch.from_numpy(np.load(a.latents or O(f"latents_{S}.npy"))).float()
    coords = torch.from_numpy(np.load(O(f"coords_{S}.npy"))).int()
    taps = {"ups": []}
    dec.z_to_f.register_forward_hook(lambda m, i, o: taps.__setitem__("z2f", o.feat.clone()))
    dec.mid.register_forward_hook(lambda m, i, o: taps.__setitem__("mid", o.feat.clone()))
    from sparc3d.models.autoencoders.blocks import SerializedUnpooling

    for name, mod in dec.named_modules():
        if isinstance(mod, SerializedUnpooling):
            def pre(m, inp):
                p = inp[0]
                taps["ups"].append([p.grid_coord.clone(), m.cls_head(p).feat.clone()])
            mod.register_forward_pre_hook(pre)
    mapping = {"grid_coord": coords, "feat": lat, "sdf": None, "deform": None, "resolution": S,
               "offset": torch.tensor([coords.shape[0]], dtype=torch.int64)}
    out = vae.decode(mapping)
    log(f"vae {S}: {out.feat.shape[0]} decoded voxels")
    save(f"vae_{S}_z2f.npy", taps["z2f"])
    save(f"vae_{S}_mid.npy", taps["mid"])
    for i, (c, l) in enumerate(taps["ups"]):
        save(f"vae_{S}_up{i}_coords.npy", c)
        save(f"vae_{S}_up{i}_logits.npy", l)
    save(f"vae_{S}_coords.npy", out.grid_coord.int())
    save(f"vae_{S}_sign.npy", out.sdf_sign)
    save(f"vae_{S}_feat.npy", out.feat)
    save(f"vae_{S}_matrix.npy", out.matrix)
    raw = point_to_mesh(out)
    save(f"mesh_{S}_raw_v.npy", raw.vertices)
    save(f"mesh_{S}_raw_f.npy", raw.faces.int())
    filled = fill_mesh_holes(raw)
    save(f"mesh_{S}_v.npy", filled.vertices)
    save(f"mesh_{S}_f.npy", filled.faces.int())
    log(f"mesh {S}: {raw.vertices.shape[0]} vertices, {raw.faces.shape[0]} faces, {filled.faces.shape[0]} filled")

# ---- post --------------------------------------------------------------------------------------------------------------
if "post" in stages:
    import trimesh

    from sparc3d.pipelines.sparc3d_pipeline import COMPONENT_FACE_RATIO, OUTPUT_RESOLUTIONS, TARGET_FACE_COUNTS
    from sparc3d.sparcubes.postprocess import postprocess_mesh

    m = trimesh.Trimesh(np.load(O(f"mesh_{S}_v.npy")), np.load(O(f"mesh_{S}_f.npy")), process=False)
    r = postprocess_mesh(m, target_face_count=TARGET_FACE_COUNTS[STRUCTURE_RESOLUTIONS.index(S)], component_face_ratio=COMPONENT_FACE_RATIO)
    save(f"final_{S}_v.npy", np.asarray(r.vertices, np.float32))
    save(f"final_{S}_f.npy", np.asarray(r.faces, np.int32))
    r.export(O(f"final_{S}.glb"))
    log(f"post: {len(r.vertices)} vertices, {len(r.faces)} faces")
