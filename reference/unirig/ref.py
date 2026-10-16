"""UniRig's reference dumps, stage by stage, for `nextsycl model3d check` (model3d/unirig/src/check.rs).

Runs the official code (github.com/VAST-AI-Research/UniRig: src/model, src/tokenizer, src/data, src/system/skin.py's
reskin) in float32 on the CPU, on a mesh read from an OBJ instead of Blender, with pure-torch stand-ins for what needs
CUDA:
- torch_cluster.fps (its CPU algorithm: start at the first point, the farthest next, distances summed x, y, z);
- spconv's SubMConv3d (submanifold: an output at every input point; the kernel's taps read the point of the neighbour
  voxel - the first point of the cloud in it when several share it - and the centre tap the point itself);
- torch_scatter.segment_csr; flash_attn's varlen attention (UniRig hands it qkv.half(): the inputs half, the product float32 here) and its MHA module;
- lightning (a plain nn.Module).
The voxel prior uses UniRig's own pyrender path (PYOPENGL_PLATFORM=egl; pyrender from /var/mnt/2TBSSD/nextsycl-ref/pyr).
transformers 4.51.3 (UniRig's) from /var/mnt/2TBSSD/nextsycl-ref/tf451.

The OBJ is Y up; Blender's importer turns it Z up ((x, y, z) -> (x, -z, y)), as UniRig extracts its meshes.

    PYTHONPATH=/var/mnt/2TBSSD/nextsycl-ref/tf451 python ref.py --repo SRC/UniRig --models ~/models/unirig \
        --mesh humanoid.obj --out ~/nsref/rig/unirig [--seed 12345]

Dumps (float32 .npy; integers as float32 where they fit, else int64 .npy):
    mesh_v, mesh_f, mesh_vn, mesh_fn       the mesh after trimesh (merged), Blender's frame
    skel_pts, skel_nrm [65536, 3]          the skeleton stage's samples (normalized)
    skel_norm [4]                          its normalization: centre (3), scale
    enc_ind [4096] (int64), enc_fps [1024] the encoder's subset and its farthest points
    enc_lat [1024, 512], cond [1024, 1024] the encoder's latents, through output_proj
    greedy [T] (int64), greedy_logits [T + 1, 267]  greedy tokens (the grammar mask, repetition penalty 3) and the
                                           logits teacher-forced on them (the prefix's last position, then each token)
    beam [T] (int64)                       the official beam sampling (15 beams, top-k 5, top-p 0.95, T 1.5)
    joints [J, 3], parents [J]             its skeleton (normalized), parents -1 for the root
    skin_norm [4], voxel_skin [V, J]       the skin stage's normalization and the voxel prior on the mesh's vertices
    skin_pts, skin_nrm [32768, 3], skin_vox [32768, J]  the skin stage's samples
    skin_joints [J, 3]                     the skeleton in the skin stage's frame
    ptv3_perm [4] (int64), ptv3_order [4, 32768] (int64)  the serialization orders as used
    skin_glob [1024, 768], skin_bone [J, 768], skin_ptv3 [32768, 768], skin_mesh [32768, 768], skin [32768, J]
    skin_v [V, J]                          reskinned onto the mesh's vertices
"""
import argparse
import contextlib
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAPTURE = {}


# ---- stand-ins ------------------------------------------------------------------------------------------------

def fps(pos, batch=None, ratio=0.5, random_start=False):
    """torch_cluster.fps on the CPU: per batch, start at its first point"""
    assert not random_start
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long)
    out = []
    for b in torch.unique(batch):
        idx = torch.nonzero(batch == b).squeeze(1)
        p = pos[idx]
        n = p.shape[0]
        k = int(np.ceil(n * ratio))
        sel = [0]

        def d2(i):
            diff = p - p[i]
            return (diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]) + diff[:, 2] * diff[:, 2]

        dist = d2(0)
        for _ in range(1, k):
            j = int(torch.argmax(dist))
            sel.append(j)
            dist = torch.minimum(dist, d2(j))
        out.append(idx[torch.tensor(sel)])
    r = torch.cat(out)
    CAPTURE.setdefault("fps", []).append(r.clone())
    return r


def segment_csr(src, indptr, reduce="sum"):
    outs = []
    for a, b in zip(indptr[:-1].tolist(), indptr[1:].tolist()):
        s = src[a:b]
        if reduce == "min":
            outs.append(s.min(0)[0])
        elif reduce == "max":
            outs.append(s.max(0)[0])
        elif reduce == "mean":
            outs.append(s.mean(0))
        else:
            outs.append(s.sum(0))
    return torch.stack(outs)


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.nbr = {}

    def replace_feature(self, f):
        t = SparseConvTensor(f, self.indices, self.spatial_shape, self.batch_size)
        t.nbr = self.nbr
        return t


class SparseModule(nn.Module):
    pass


class SubMConv3d(SparseModule):
    def __init__(self, cin, cout, kernel_size=3, padding=0, bias=True, indice_key=None):
        super().__init__()
        self.k = kernel_size
        self.weight = nn.Parameter(torch.zeros(cout, kernel_size, kernel_size, kernel_size, cin))
        self.bias = nn.Parameter(torch.zeros(cout)) if bias else None

    def neighbours(self, t):
        key = self.k
        if key not in t.nbr:
            ind = t.indices[:, 1:].long()
            first = {}
            for i, c in enumerate(map(tuple, ind.tolist())):
                first.setdefault(c, i)
            r = self.k // 2
            offs = [(a, b, c) for a in range(-r, r + 1) for b in range(-r, r + 1) for c in range(-r, r + 1)]
            nb = torch.full((ind.shape[0], len(offs)), -1, dtype=torch.long)
            for i, c in enumerate(map(tuple, ind.tolist())):
                for o, (a, b, cc) in enumerate(offs):
                    if a == 0 and b == 0 and cc == 0:
                        nb[i, o] = i
                    else:
                        nb[i, o] = first.get((c[0] + a, c[1] + b, c[2] + cc), -1)
            t.nbr[key] = nb
        return t.nbr[key]

    def forward(self, t):
        nb = self.neighbours(t)
        f = t.features
        n, cin = f.shape
        fz = torch.cat([f, torch.zeros(1, cin, dtype=f.dtype)])
        cols = fz[nb.clamp(min=-1) % (n + 1)]                      # [N, K, Cin], -1 -> the zero row
        w = self.weight.reshape(self.weight.shape[0], -1)          # [Cout, K * Cin]
        out = cols.reshape(n, -1) @ w.t()
        if self.bias is not None:
            out = out + self.bias
        return t.replace_feature(out)


def is_spconv_module(m):
    return isinstance(m, SparseModule)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None):
    """the product in float32 on the half inputs UniRig hands it, per patch"""
    qkv = qkv.float()
    out = torch.empty(qkv.shape[0], qkv.shape[2], qkv.shape[3])
    cu = cu_seqlens.tolist()
    for a, b in zip(cu[:-1], cu[1:]):
        q, k, v = qkv[a:b, 0].transpose(0, 1), qkv[a:b, 1].transpose(0, 1), qkv[a:b, 2].transpose(0, 1)
        out[a:b] = F.scaled_dot_product_attention(q, k, v, scale=softmax_scale).transpose(0, 1)
    return out


class MHA(nn.Module):
    """flash_attn.modules.mha.MHA, cross attention only: Wq, Wkv ((two h d) packed), out_proj"""

    def __init__(self, embed_dim, num_heads, cross_attn=True, **kw):
        super().__init__()
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_kv=None, **kw):
        B, N, C = x.shape
        q = self.Wq(x).view(B, N, self.h, self.d).transpose(1, 2)
        kv = self.Wkv(x_kv).view(B, x_kv.shape[1], 2, self.h, self.d)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(o)


def install():
    torch.backends.cuda.sdp_kernel = lambda *a, **k: contextlib.nullcontext()
    m = types.ModuleType("torch_cluster")
    m.fps = fps
    sys.modules["torch_cluster"] = m
    m = types.ModuleType("torch_scatter")
    m.segment_csr = segment_csr
    sys.modules["torch_scatter"] = m
    sp = types.ModuleType("spconv")
    spp = types.ModuleType("spconv.pytorch")
    spp.SparseConvTensor = SparseConvTensor
    spp.SubMConv3d = SubMConv3d
    mods = types.ModuleType("spconv.pytorch.modules")
    mods.is_spconv_module = is_spconv_module
    mods.SparseModule = SparseModule
    spp.modules = mods
    sp.pytorch = spp
    sys.modules.update({"spconv": sp, "spconv.pytorch": spp, "spconv.pytorch.modules": mods})
    fa = types.ModuleType("flash_attn")
    fa.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    fam = types.ModuleType("flash_attn.modules")
    famh = types.ModuleType("flash_attn.modules.mha")
    famh.MHA = MHA
    fa.modules = fam
    fam.mha = famh
    sys.modules.update({"flash_attn": fa, "flash_attn.modules": fam, "flash_attn.modules.mha": famh})
    lt = types.ModuleType("lightning")
    ltp = types.ModuleType("lightning.pytorch")
    ltp.LightningModule = nn.Module
    cb = types.ModuleType("lightning.pytorch.callbacks")
    cb.BasePredictionWriter = object
    ut = types.ModuleType("lightning.pytorch.utilities")
    ut.grad_norm = None
    lt.pytorch = ltp
    lt.LightningModule = nn.Module
    sys.modules.update({"lightning": lt, "lightning.pytorch": ltp, "lightning.pytorch.callbacks": cb, "lightning.pytorch.utilities": ut})
    # importlib looks at a module's spec (transformers asks whether flash_attn is there: it is not a package)
    import importlib.machinery
    for name in ["torch_cluster", "torch_scatter", "spconv", "spconv.pytorch", "spconv.pytorch.modules", "flash_attn", "flash_attn.modules",
                 "flash_attn.modules.mha", "lightning", "lightning.pytorch", "lightning.pytorch.callbacks", "lightning.pytorch.utilities"]:
        sys.modules[name].__spec__ = importlib.machinery.ModuleSpec(name, None)


# ---- helpers -------------------------------------------------------------------------------------------------

def save(out, name, x, ints=False):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(x.astype(np.int64 if ints else np.float32)))


def read_obj(path):
    v, f = [], []
    for line in open(path):
        s = line.split()
        if not s:
            continue
        if s[0] == "v":
            v.append([float(x) for x in s[1:4]])
        elif s[0] == "f":
            c = [int(x.split("/")[0]) - 1 for x in s[1:]]
            for k in range(1, len(c) - 1):
                f.append([c[0], c[k], c[k + 1]])
    v = np.array(v, dtype=np.float32)
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)   # Blender's frame
    return v, np.array(f, dtype=np.int64)


def normalization(points):
    """AugmentAffine: centre on the box, scale its longest side into [-1, 1]: (centre, scale)"""
    lo, hi = points.min(0), points.max(0)
    return (hi + lo) / 2, np.max((hi - lo) / 2.0)


def load_state(model, ckpt, prefix="model."):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    r = model.load_state_dict(sd, strict=False)
    print("load", ckpt, "missing", r.missing_keys[:6], "unexpected", r.unexpected_keys[:6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--stages", default="skeleton,skin")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    sys.path.append("/var/mnt/2TBSSD/nextsycl-ref/pyr")
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    install()
    import trimesh
    import yaml
    from box import Box
    from transformers import AutoConfig, OPTConfig
    from src.tokenizer.spec import TokenizerConfig
    from src.tokenizer.parse import get_tokenizer
    from src.model.parse import get_model
    from src.data.sampler import SamplerMix, SamplerConfig
    from src.data.asset import Asset
    from src.data.vertex_group import VertexGroupVoxelSkin
    from src.system.skin import reskin

    # OPT-350m's config as UniRig changes it (no download)
    def opt_config(**kw):
        c = OPTConfig(vocab_size=kw.get("vocab_size", 50272), hidden_size=1024, num_hidden_layers=24, ffn_dim=4096, num_attention_heads=16,
                      max_position_embeddings=kw.get("max_position_embeddings", 2048), word_embed_proj_dim=1024, do_layer_norm_before=True,
                      activation_function="relu", dropout=0.0, attention_dropout=0.0, enable_bias=True, layer_norm_elementwise_affine=True)
        c._attn_implementation = "eager"
        return c
    AutoConfig.from_pretrained = staticmethod(opt_config)

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    # the mesh as UniRig's extract leaves it (trimesh merges, normals)
    v, f = read_obj(a.mesh)
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mv = np.array(mesh.vertices, dtype=np.float32)
    mf = np.array(mesh.faces, dtype=np.int64)
    mvn = np.array(mesh.vertex_normals, dtype=np.float32)
    mfn = np.array(mesh.face_normals, dtype=np.float32)
    save(a.out, "mesh_v", mv)
    save(a.out, "mesh_f", mf)
    save(a.out, "mesh_vn", mvn)
    save(a.out, "mesh_fn", mfn)

    def asset_of(verts, joints=None):
        return Asset(cls=None, path="", data_name="", vertices=verts.copy(), vertex_normals=mvn.copy(), faces=mf.copy(), face_normals=mfn.copy(),
                     joints=joints, tails=None, skin=None, no_skin=None, parents=None, names=None, matrix_local=None)

    tok_cfg = Box(yaml.safe_load(open("configs/tokenizer/tokenizer_parts_articulationxl_256.yaml")))
    tokenizer = get_tokenizer(config=TokenizerConfig.parse(config=tok_cfg))

    if "skeleton" in a.stages:
        c, s = normalization(mv)
        save(a.out, "skel_norm", np.array([c[0], c[1], c[2], s]))
        nv = ((mv - c) / s).astype(np.float32)
        asset = asset_of(nv)
        asset.vertex_groups = {}
        samp = SamplerMix(SamplerConfig(method="mix", num_samples=65536, vertex_samples=8192, kwargs={})).sample(asset)
        pts = torch.tensor(samp.vertices, dtype=torch.float32)
        nrm = torch.tensor(samp.normals, dtype=torch.float32)
        save(a.out, "skel_pts", pts)
        save(a.out, "skel_nrm", nrm)
        mcfg = Box(yaml.safe_load(open("configs/model/unirig_ar_350m_1024_81920_float32.yaml")))
        model = get_model(tokenizer=tokenizer, **mcfg)
        load_state(model, os.path.join(a.models, "skeleton/articulation-xl_quantization_256/model.ckpt"))
        model.eval()
        with torch.no_grad():
            rng = np.random.default_rng(seed=0)
            ind = rng.choice(pts.shape[0], 4096, replace=4096 > pts.shape[0])
            save(a.out, "enc_ind", ind, ints=True)
            CAPTURE.clear()
            _, lat, _, _ = model.mesh_encoder.encode_latents(pc=pts[None], feats=nrm[None])
            save(a.out, "enc_fps", CAPTURE["fps"][0], ints=True)
            save(a.out, "enc_lat", lat[0])
            cond = model.output_proj(lat)
            save(a.out, "cond", cond[0])
            start = [tokenizer.bos, tokenizer.cls_name_to_token(cls="articulationxl")]
            # greedy with the grammar and the repetition penalty
            torch.manual_seed(a.seed)
            g = model.generate(vertices=pts, normals=nrm, cls="articulationxl", max_new_tokens=a.max_new, num_beams=1, do_sample=False,
                               repetition_penalty=3.0)
            ids = np.array(g.tokens)
            print("greedy: %d tokens, %d joints" % (len(ids), len(g.bones)))
            # the generated tokens after the start ones (detokenize's input had bos .. eos stripped)
            gen = model.transformer.generate(inputs_embeds=torch.cat([cond, model.transformer.get_input_embeddings()(torch.tensor([start]))], 1),
                                             bos_token_id=tokenizer.bos, eos_token_id=tokenizer.eos, pad_token_id=tokenizer.pad,
                                             logits_processor=__import__("transformers").LogitsProcessorList(
                                                 [__import__("src.model.unirig_ar", fromlist=["x"]).VocabSwitchingLogitsProcessor(tokenizer, torch.tensor(start))]),
                                             max_new_tokens=a.max_new, num_beams=1, do_sample=False, repetition_penalty=3.0)[0]
            save(a.out, "greedy", gen, ints=True)
            emb = model.transformer.get_input_embeddings()(torch.tensor([start + gen.tolist()]))
            out = model.transformer(inputs_embeds=torch.cat([cond, emb], 1))
            save(a.out, "greedy_logits", out.logits[0, cond.shape[1] + len(start) - 1:])
            # the official sampling
            torch.manual_seed(a.seed)
            res = model.generate(vertices=pts, normals=nrm, cls="articulationxl", max_new_tokens=a.max_new, num_return_sequences=1, num_beams=15,
                                 do_sample=True, top_k=5, top_p=0.95, repetition_penalty=3.0, temperature=1.5)
            save(a.out, "beam", np.array(res.tokens), ints=True)
            joints = np.array(res.bones[:, 3:], dtype=np.float32)
            parents = np.array([-1 if p is None else p for p in res.parents])
            save(a.out, "joints", joints)
            save(a.out, "parents", parents)
            print("beam sampling: %d tokens, %d joints" % (len(res.tokens), len(joints)))
            # back to the mesh's frame for the skin stage
            np.save(os.path.join(a.out, "joints_mesh.npy"), (joints * s + c).astype(np.float32))

    if "skin" in a.stages:
        joints_mesh = np.load(os.path.join(a.out, "joints_mesh.npy"))
        parents = np.load(os.path.join(a.out, "parents.npy")).astype(np.int64)
        J = len(joints_mesh)
        c, s = normalization(np.concatenate([mv, joints_mesh]))
        save(a.out, "skin_norm", np.array([c[0], c[1], c[2], s]))
        nv = ((mv - c) / s).astype(np.float32)
        nj = ((joints_mesh - c) / s).astype(np.float32)
        save(a.out, "skin_joints", nj)
        asset = asset_of(nv, joints=nj)
        vg = VertexGroupVoxelSkin(grid=196, alpha=0.5, link_dis=0.00001, grid_query=7, vertex_query=1, grid_weight=3.0, backend="pyrender")
        vox = vg.get_vertex_group(asset)["voxel_skin"]
        save(a.out, "voxel_skin", vox)
        asset.vertex_groups = {"voxel_skin": vox}
        samp = SamplerMix(SamplerConfig(method="mix", num_samples=32768, vertex_samples=8192, kwargs={})).sample(asset)
        pts = torch.tensor(samp.vertices, dtype=torch.float32)
        nrm = torch.tensor(samp.normals, dtype=torch.float32)
        sv = torch.tensor(samp.vertex_groups["voxel_skin"], dtype=torch.float32)
        save(a.out, "skin_pts", pts)
        save(a.out, "skin_nrm", nrm)
        save(a.out, "skin_vox", sv)
        mcfg = Box(yaml.safe_load(open("configs/model/unirig_skin.yaml")))
        model = get_model(**mcfg)
        load_state(model, os.path.join(a.models, "skin/articulation-xl/model.ckpt"))
        model.eval()
        hooks = []

        def keep(name):
            def h(m, i, o):
                CAPTURE[name] = (o.feat if hasattr(o, "feat") else o).detach().clone()
            return h
        hooks.append(model.out_proj.register_forward_hook(keep("glob")))
        hooks.append(model.bone_encoder.register_forward_hook(keep("bone")))
        hooks.append(model.feat_map.register_forward_hook(keep("ptv3")))
        hooks.append(model.mesh_bone_attn[-1].register_forward_hook(keep("mesh")))
        real_ser = __import__("src.model.pointcept.models.utils.structure", fromlist=["Point"]).Point.serialization

        def ser(self, *ar, **kw):
            real_ser(self, *ar, **kw)
            CAPTURE["order"] = self["serialized_order"].clone()
        __import__("src.model.pointcept.models.utils.structure", fromlist=["Point"]).Point.serialization = ser
        real_randperm = torch.randperm

        def randperm(n, *ar, **kw):
            p = real_randperm(n, *ar, **kw)
            if n == 4:
                CAPTURE["perm"] = p.clone()
            return p
        torch.randperm = randperm
        batch = {
            "num_bones": torch.tensor([J]), "vertices": pts[None], "normals": nrm[None], "joints": torch.tensor(nj)[None],
            "tails": torch.tensor(nj)[None], "voxel_skin": sv[None], "parents": torch.tensor(parents)[None], "offset": [pts.shape[0]],
        }
        with torch.no_grad():
            CAPTURE.clear()
            rng = np.random.default_rng(seed=0)
            save(a.out, "skin_enc_ind", rng.choice(pts.shape[0], 4096, replace=4096 > pts.shape[0]), ints=True)
            skin, _ = model._get_predict(batch)
            save(a.out, "skin_enc_fps", CAPTURE["fps"][0], ints=True)
        torch.randperm = real_randperm
        save(a.out, "ptv3_perm", CAPTURE["perm"], ints=True)
        save(a.out, "ptv3_order", CAPTURE["order"], ints=True)
        save(a.out, "skin_glob", CAPTURE["glob"][0])
        save(a.out, "skin_bone", CAPTURE["bone"][0])
        save(a.out, "skin_ptv3", CAPTURE["ptv3"].reshape(-1, CAPTURE["ptv3"].shape[-1]))
        save(a.out, "skin_mesh", CAPTURE["mesh"][0])
        save(a.out, "skin", skin[0])
        pl = [None if p < 0 else int(p) for p in parents]
        sk = reskin(sampled_vertices=pts.numpy(), vertices=nv, parents=pl, faces=mf, sampled_skin=skin[0].numpy(), sample_method="median", alpha=2.0,
                    threshold=0.03)
        save(a.out, "skin_v", sk)
        print("skin: %d points, %d joints; mesh %d vertices" % (pts.shape[0], J, len(nv)))


if __name__ == "__main__":
    main()
