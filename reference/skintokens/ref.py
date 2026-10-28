"""SkinTokens / TokenRig reference dumps, stage by stage, for `nextsycl model3d check` (model3d/skintokens/src/check.rs).

The model's own modules (github.com/VAST-AI-Research/SkinTokens @ 273b691: src/model/tokenrig.py, src/model/michelangelo,
src/model/skin_vae, src/tokenizer/tokenizer_part.py, src/data/sampler.py + src/rig_package/utils.py) TRANSCRIBED here in
float32 on the CPU: the repository's code is never imported or run (it needs bpy, flash-attn and CUDA, and is not ours to
execute), and the checkpoints are read by `SafeCkpt` below - a pickle reader that allows only OrderedDict, torch's tensor
rebuild and the storage types, and refuses everything else (never torch.load). The one library piece used as it is: the
Qwen3 decoder (transformers' Qwen3ForCausalLM, built from the config the checkpoint names - the model is built from
scratch, no Qwen weights) and its `generate` for the greedy reference.

The mesh is read from an OBJ instead of Blender (Y up; Blender's importer turns it Z up: (x, y, z) -> (x, -z, y)), and
the demo's path is followed: a trimesh with process=False (no merging), normalized into [-1, 1] by its box (AugmentAffine,
AugmentNormalize), 54,000 samples (16,384 of them vertices; SamplerMix), the shape encoder (ShapeAsLatentPerceiverEncoder),
the skin VAE's condition encoder (384 tokens), the decoder over [shape latents, bos, class articulation] with the
VocabSwitchingLogitsProcessor grammar, the skin tokens through the FSQ and the VAE decoder.

Two details of the reference worth saying: the output projection's RMSNorm is torch's nn.RMSNorm(eps=None), whose eps is
the dtype's machine epsilon - the model runs in bfloat16, so 2^-7 (used here); and the skin switch forces the closing
eos after J * 4 - 1 skin tokens (length - where == J * 4 counts the switch token), so the last bone's fourth token reads
the eos id as a code (index 32768 -> level indices all 0, FSQ code index 0) - reproduced as the demo has it.

    python -I ref.py --models /var/mnt/2TBSSD/skintokens-models --mesh humanoid.obj --out ~/nsref/rig/skintokens [--seed 12345]

Dumps (float32 .npy; integers int64):
    mesh_v, mesh_f, mesh_vn      the mesh as read (Blender's frame), its trimesh vertex normals
    mesh_vnorm [V, 3]            its vertices normalized into [-1, 1]; norm [4]: centre (3), scale
    samp_pts, samp_nrm [54000, 3]  the samples (normalized frame), normals unit
    enc_ind [2048] (int64), enc_fps [512]  the shape encoder's random subset and its farthest points (indices into the subset)
    enc_lat [512, 512], cond [512, 896]    its latents, through output_proj
    vae_ind [1536] (int64), vae_fps [384]  the skin VAE's condition encoder's subset and farthest points
    cond_lat [384, 512]                    its tokens (cond_quant)
    greedy [T] (int64), greedy_logits [T + 1, 33036]  greedy tokens (grammar, repetition penalty 2) and the logits
                                           teacher-forced on them (the prefix's last position, then each token)
    joints [J, 3], parents [J]             the skeleton (normalized frame), parents -1 for the root
    fsq_z [J, 4, 512]                      the skin tokens through the FSQ's codes and project_out
    dec_kv0 [388, 768]                     bone 0's decoder after its self-attention blocks
    skin [54000, J]                        the VAE decoder's weights at the samples
    skin_v [V, J], weights [V, J]          carried to the mesh's vertices (8 nearest samples, inverse distance), then the 4
                                           largest of a vertex renormalized
"""
import argparse
import io
import math
import os
import pickle
import zipfile
import collections

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)


# ---- a checkpoint without torch.load ----------------------------------------------------------------------------

class _Storage:
    def __init__(self, kind, key):
        self.kind, self.key = kind, key


_DT = {"BFloat16Storage": torch.bfloat16, "FloatStorage": torch.float32, "HalfStorage": torch.float16, "LongStorage": torch.int64}


class _Lazy:
    """a tensor in the zip: read when asked"""

    def __init__(self, st, off, size, stride):
        self.st, self.off, self.size, self.stride = st, off, tuple(size), tuple(stride)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda st, off, size, stride, *a: _Lazy(st, off, size, stride)
        if module == "torch" and name in _DT:
            return name
        raise pickle.UnpicklingError(f"refused {module}.{name}")

    def persistent_load(self, pid):
        return _Storage(pid[1], pid[2])


class SafeCkpt:
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        pk = [n for n in self.z.namelist() if n.endswith("data.pkl")][0]
        self.root = pk[: -len("data.pkl")]
        o = _Unpickler(io.BytesIO(self.z.read(pk))).load()
        self.hp = o["hyper_parameters"]
        self.lazy = o["state_dict"]
        self.cache = {}

    def get(self, name):
        l = self.lazy[name]
        raw = self.z.read(f"{self.root}data/{l.st.key}")
        t = torch.frombuffer(bytearray(raw), dtype=_DT[l.st.kind])
        n = int(np.prod(l.size)) if l.size else 1
        return t.as_strided(l.size, l.stride, l.off).to(torch.float32).clone() if n else torch.zeros(l.size)

    def state(self):
        return {k: self.get(k) for k in self.lazy}


# ---- the tokenizer (src/tokenizer/tokenizer_part.py) --------------------------------------------------------------

class Tok:
    def __init__(self, cfg):
        self.num_discrete = cfg["num_discrete"]
        self.lo, self.hi = cfg["continuous_range"]
        o = self.num_discrete
        self.branch, self.bos, self.eos, self.pad = o, o + 1, o + 2, o + 3
        o += 4
        self.spring = o
        o += 1
        self.parts = {k: v + o for k, v in cfg["parts_token_id"].items()}
        o += len(self.parts)
        self.cls_none = o
        o += 1
        self.cls = {k: v + o for k, v in cfg["cls_token_id"].items()}
        o += len(self.cls)
        self.vocab_size = o

    def next_possible(self, ids):
        if len(ids) == 0:
            return [self.bos]
        st = "expect_bos"
        for i in ids:
            if st == "expect_bos":
                assert i == self.bos
                st = "expect_cls_or_part_or_joint"
            elif st == "expect_cls_or_part_or_joint":
                if i < self.num_discrete:
                    st = "expect_joint_2"
                elif i == self.cls_none or i in self.cls.values():
                    st = "expect_part_or_joint"
                else:
                    st = "expect_joint"
            elif st == "expect_part_or_joint":
                st = "expect_joint_2" if i < self.num_discrete else "expect_part_or_joint"
            elif st == "expect_joint_2":
                st = "expect_joint_3"
            elif st == "expect_joint_3":
                st = "expect_branch_or_part_or_joint"
            elif st == "expect_branch_or_part_or_joint":
                if i == self.branch:
                    st = "expect_joint"
                elif i < self.num_discrete:
                    st = "expect_joint_2"
                else:
                    st = "expect_joint"
            elif st == "expect_joint":
                st = "expect_joint_2"
        s = []
        cls = lambda: s.extend([self.cls_none] + list(self.cls.values()))
        part = lambda: s.extend([self.spring] + list(self.parts.values()))
        joint = lambda: s.extend(range(self.num_discrete))
        if st == "expect_cls_or_part_or_joint":
            cls(); part(); joint()
        elif st == "expect_part_or_joint":
            part(); joint(); s.append(self.eos)
        elif st in ("expect_joint_2", "expect_joint_3", "expect_joint"):
            joint()
        elif st == "expect_branch_or_part_or_joint":
            joint(); part(); s.append(self.branch); s.append(self.eos)
        else:
            raise AssertionError(st)
        return s

    def bones_in_sequence(self, ids):
        s = 0
        is_branch = False
        st = "expect_bos"
        for i in ids:
            if st == "expect_bos":
                st = "expect_cls_or_part_or_joint"
            elif st == "expect_cls_or_part_or_joint":
                if i < self.num_discrete:
                    st = "expect_joint_2"
                elif i == self.cls_none or i in self.cls.values():
                    st = "expect_part_or_joint"
                else:
                    st = "expect_joint"
            elif st == "expect_part_or_joint":
                st = "expect_joint_2" if i < self.num_discrete else "expect_part_or_joint"
            elif st == "expect_joint_2":
                st = "expect_joint_3"
            elif st == "expect_joint_3":
                if not is_branch:
                    s += 1
                is_branch = False
                st = "expect_branch_or_part_or_joint"
            elif st == "expect_branch_or_part_or_joint":
                if i == self.branch:
                    st = "expect_joint"
                    is_branch = True
                elif i < self.num_discrete:
                    st = "expect_joint_2"
                else:
                    st = "expect_joint"
            elif st == "expect_joint":
                st = "expect_joint_2"
            if i == self.eos:
                break
        return s

    def undiscretize(self, t):
        t = np.asarray(t).astype(np.float32) + 0.5
        t /= self.num_discrete
        return t * (self.hi - self.lo) + self.lo

    def detokenize(self, ids):
        """joints and parents (make_skeleton with leaf and branch tails: only the parents and joints are used here)"""
        assert ids[0] == self.bos
        t = 0
        while ids[-1 - t] == self.pad:
            t += 1
        assert ids[-1 - t] == self.eos
        ids = ids[1:-1 - t]
        joints, p_joints = [], []
        i, is_branch, last = 0, False, None
        while i < len(ids):
            if ids[i] < self.num_discrete:
                if is_branch:
                    p_joints.append(self.undiscretize(ids[i:i + 3]))
                    joints.append(self.undiscretize(ids[i + 3:i + 6]))
                    i += 6
                else:
                    cur = self.undiscretize(ids[i:i + 3])
                    joints.append(cur)
                    p_joints.append(cur if len(p_joints) == 0 else last)
                    i += 3
                last = joints[-1]
                is_branch = False
            elif ids[i] == self.branch:
                is_branch, last = True, None
                i += 1
            else:
                i += 1  # a part or class token
        joints = np.stack(joints)
        parents = [-1]
        for k in range(1, len(joints)):
            dis, pid = 999999, None
            for j in reversed(range(k)):
                d = ((joints[j] - p_joints[k]) ** 2).sum()
                if d < dis:
                    pid, dis = j, d
            parents.append(pid)
        return joints.astype(np.float32), np.array(parents)


# ---- the sampler (src/data/sampler.py, src/rig_package/utils.py) ---------------------------------------------------

def sample_surface(num, v, f):
    off0, off1 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    w = np.linalg.norm(np.cross(off0, off1, axis=-1), axis=-1)
    cum = np.cumsum(w, axis=0)
    pick = np.random.rand(num) * cum[-1]
    fi = np.searchsorted(cum, pick)
    org = v[f[:, 0]]
    vec = v[f[:, 1:]]
    vec -= np.tile(org, (1, 2)).reshape((-1, 2, 3))
    org, vec = org[fi], vec[fi]
    rl = np.random.rand(len(vec), 2, 1)
    t = rl.sum(axis=1).reshape(-1) > 1.0
    rl[t] -= 1.0
    rl = np.abs(rl)
    return (vec * rl).sum(axis=1) + org, fi


def sample_mix(v, f, vn, fn, num_samples, num_vertex_samples):
    perm = np.random.permutation(v.shape[0])[:num_vertex_samples]
    fv, fi = sample_surface(num_samples - len(perm), v, f)
    return np.concatenate([v[perm], fv]), np.concatenate([vn[perm], fn[fi]])


# ---- the shape encoder (src/model/michelangelo, ShapeAsLatentPerceiverEncoder; width 512, 8 heads, 8 layers) -------

def ln(x, w, b, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def fps_torch(x, ratio):
    """src/model/utils.py fps, one batch, not random_start"""
    n = x.shape[0]
    k = min(max(1, int(round(n * ratio))), n)
    d = torch.full((n,), float("inf"))
    sel = torch.empty(k, dtype=torch.long)
    far = 0
    for i in range(k):
        sel[i] = far
        d = torch.minimum(d, torch.sum((x - x[far]) ** 2, dim=-1))
        far = int(torch.argmax(d))
    return sel


def fourier(x, num_freqs=8, include_pi=False):
    fr = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
    if include_pi:
        fr = fr * torch.pi
    e = (x[..., None].contiguous() * fr).view(*x.shape[:-1], -1)
    return torch.cat((x, e.sin(), e.cos()), dim=-1)


def lin(x, sd, p, bias=True):
    return F.linear(x, sd[p + ".weight"], sd[p + ".bias"] if bias else None)


def mlp(x, sd, p):
    return lin(F.gelu(lin(x, sd, p + ".c_fc")), sd, p + ".c_proj")


def mha_self(x, sd, p, heads):
    qkv = F.linear(x, sd[p + ".c_qkv.weight"])
    b, n, w = qkv.shape
    qkv = qkv.view(b, n, heads, -1)
    ch = w // heads // 3
    q, k, v = torch.split(qkv, ch, dim=-1)
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).reshape(b, n, -1)
    return lin(o, sd, p + ".c_proj")


def mha_cross(x, data, sd, p, heads):
    q = F.linear(x, sd[p + ".c_q.weight"])
    kv = F.linear(data, sd[p + ".c_kv.weight"])
    b, nq, w = q.shape
    nd = kv.shape[1]
    q = q.view(b, nq, heads, -1)
    kv = kv.view(b, nd, heads, -1)
    k, v = torch.split(kv, kv.shape[-1] // 2, dim=-1)
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).reshape(b, nq, -1)
    return lin(o, sd, p + ".c_proj")


def mesh_encode(sd, pc, feats, ind):
    """pc, feats [1, N, 3]; ind the random subset: latents [1, 512, 512], the farthest points (indices into the subset)"""
    p = "mesh_encoder.encoder"
    pre_pc, pre_f = pc[:, ind, :], feats[:, ind, :]
    sel = fps_torch(pre_pc[0], 0.25)
    sampled = pre_pc[:, sel]
    sampled_f = pre_f[:, sel]
    data = lin(torch.cat([fourier(pc), feats], -1), sd, p + ".input_proj")
    q = lin(torch.cat([fourier(sampled), sampled_f], -1), sd, p + ".input_proj")
    c = p + ".cross_attn"
    x = q + mha_cross(ln(q, sd[c + ".ln_1.weight"], sd[c + ".ln_1.bias"]), ln(data, sd[c + ".ln_2.weight"], sd[c + ".ln_2.bias"]), sd, c + ".attn", 8)
    x = x + mlp(ln(x, sd[c + ".ln_3.weight"], sd[c + ".ln_3.bias"]), sd, c + ".mlp")
    for i in range(8):
        r = f"{p}.self_attn.resblocks.{i}"
        x = x + mha_self(ln(x, sd[r + ".ln_1.weight"], sd[r + ".ln_1.bias"]), sd, r + ".attn", 8)
        x = x + mlp(ln(x, sd[r + ".ln_2.weight"], sd[r + ".ln_2.bias"]), sd, r + ".mlp")
    x = ln(x, sd[p + ".ln_post.weight"], sd[p + ".ln_post.bias"])
    return x, sel


# ---- the skin VAE (src/model/skin_vae: Tripo2Encoder / Tripo2Decoder, DiTBlock, FSQ; width 768, 12 heads) ------------

def pmpe_embed(x, num_freqs=8):
    """FrequencyPositionalEmbedding(num_freqs=8, logspace, include_pi=True, use_pmpe=True), include_input"""
    fr = (2.0 ** torch.arange(num_freqs, dtype=torch.float32)) * torch.pi
    phase = torch.arange(num_freqs, dtype=torch.float32)
    for i in range(num_freqs):
        phase[i] = torch.pow(torch.tensor(num_freqs), 1.0 - (i + 1) / num_freqs) + (i + 1) / num_freqs
    phase *= torch.pi * 2
    embed = (x[..., None].contiguous() * fr).view(*x.shape[:-1], -1)
    ph = (x[..., None].contiguous() * torch.pi * 0.5 + phase).view(*x.shape[:-1], -1)
    res = torch.cat((embed.sin() + ph.sin(), embed.cos() + ph.cos()), dim=-1)
    return torch.cat((x, res), dim=-1)


def attn_dit(x, enc, sd, p, heads, cross):
    """diffusers Attention with the Tripo2 processor: q, k, v without bias; self: [q | k | v] split per head as (heads, 3 hd);
    cross: [k | v] split per head as (heads, 2 hd); to_out with bias"""
    b = x.shape[0]
    q = F.linear(x, sd[p + ".to_q.weight"])
    if cross:
        e = ln(enc, sd[p + ".norm_cross.weight"], sd[p + ".norm_cross.bias"])
    else:
        e = x
    k = F.linear(e, sd[p + ".to_k.weight"])
    v = F.linear(e, sd[p + ".to_v.weight"])
    if not cross:
        qkv = torch.cat((q, k, v), dim=-1)
        sp = qkv.shape[-1] // heads // 3
        qkv = qkv.view(b, -1, heads, sp * 3)
        q, k, v = torch.split(qkv, sp, dim=-1)
    else:
        kv = torch.cat((k, v), dim=-1)
        sp = kv.shape[-1] // heads // 2
        kv = kv.view(b, -1, heads, sp * 2)
        k, v = torch.split(kv, sp, dim=-1)
    hd = k.shape[-1]
    q = q.reshape(b, -1, heads, hd).transpose(1, 2)
    k = k.reshape(b, -1, heads, hd).transpose(1, 2)
    v = v.reshape(b, -1, heads, hd).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, -1, heads * hd)
    return lin(o, sd, p + ".to_out.0")


def dit_block(x, sd, p, heads, enc=None):
    """DiTBlock: self-attention (norm1, attn1) or cross-attention (norm2, attn2), then the feed-forward (norm3, GELU)"""
    if enc is None:
        x = x + attn_dit(ln(x, sd[p + ".norm1.weight"], sd[p + ".norm1.bias"]), None, sd, p + ".attn1", heads, False)
    else:
        x = x + attn_dit(ln(x, sd[p + ".norm2.weight"], sd[p + ".norm2.bias"]), enc, sd, p + ".attn2", heads, True)
    h = ln(x, sd[p + ".norm3.weight"], sd[p + ".norm3.bias"])
    h = F.gelu(lin(h, sd, p + ".ff.net.0.proj"))
    return x + lin(h, sd, p + ".ff.net.2")


def vae_cond_encode(sd, cond, ind):
    """SkinFSQCVAEModel._encode (return_z False): cond [1, N, 6] -> [1, 384, 512]"""
    p = "vae.model.cond_encoder"
    pos, feat = cond[..., :3], cond[..., 3:]
    kv = torch.cat([pmpe_embed(pos), feat], -1)
    sub = cond[:, ind]
    sel = fps_torch(sub[0, :, :3], 0.25)
    s = sub[:, sel]
    q = torch.cat([pmpe_embed(s[..., :3]), s[..., 3:]], -1)
    h = lin(q, sd, p + ".proj_in")
    e = lin(kv, sd, p + ".proj_in")
    h = dit_block(h, sd, p + ".blocks.0", 12, enc=e)
    for i in (1, 2):
        h = dit_block(h, sd, f"{p}.blocks.{i}", 12)
    h = ln(h, sd[p + ".norm_out.weight"], sd[p + ".norm_out.bias"])
    return lin(h, sd, "vae.model.cond_quant"), sel


LEVELS = [8, 8, 8, 8, 8]


def fsq_codes(sd, idx):
    """FSQ.indices_to_codes: indices [n] -> project_out of the level codes [n, 512]"""
    levels = torch.tensor(LEVELS, dtype=torch.int32)
    basis = torch.cumprod(torch.tensor([1] + LEVELS[:-1]), dim=0, dtype=torch.int32)
    li = (idx.to(torch.int32)[..., None] // basis) % levels
    half = levels // 2
    codes = (li - half) / half
    return lin(codes.to(torch.float32), sd, "vae.model.FSQ.project_out")


def vae_decode(sd, z, cond_lat, pts, chunk=6000):
    """SkinFSQCVAEModel._decode for one bone: z [1, 4, 512], cond_lat [1, 384, 512], pts [1, N, 6] -> [N]; also the kv_cache"""
    x = lin(torch.cat([z, cond_lat], dim=1), sd, "vae.model.post_quant")
    for i in range(10):
        x = dit_block(x, sd, f"vae.model.decoder.blocks.{i}", 12)
    kv = x
    out = []
    for s in range(0, pts.shape[1], chunk):
        pc = pts[:, s:s + chunk]
        q = lin(torch.cat([pmpe_embed(pc[..., :3]), pc[..., 3:]], -1), sd, "vae.model.decoder.proj_query")
        l = dit_block(q, sd, "vae.model.decoder.blocks.10", 12, enc=kv)
        l = lin(ln(l, sd["vae.model.decoder.norm_out.weight"], sd["vae.model.decoder.norm_out.bias"]), sd, "vae.model.decoder.proj_out")
        out.append(torch.sigmoid(l)[0, :, 0])
    return torch.cat(out), kv[0]


# ---- main ------------------------------------------------------------------------------------------------------

def read_obj(path):
    v, f = [], []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            v.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            ix = [int(x.split("/")[0]) - 1 for x in t[1:]]
            for k in range(1, len(ix) - 1):
                f.append([ix[0], ix[k], ix[k + 1]])
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--stages", default="encode,tokens,skin")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    stages = a.stages.split(",")

    def dump(name, x, dt=None):
        x = np.asarray(x.numpy() if torch.is_tensor(x) else x)
        x = x.astype(dt or (np.int64 if x.dtype.kind in "iu" else np.float32))
        np.save(os.path.join(a.out, name + ".npy"), x)

    import trimesh
    from scipy.spatial import cKDTree
    ck = SafeCkpt(os.path.join(a.models, "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"))
    hp = ck.hp
    tok = Tok(hp["tokenizer_config"])
    mc = hp["model_config"]
    assert (tok.vocab_size, mc["tokens_per_skin"], mc["tokens_skin_cond"], mc["llm"]["hidden_size"]) == (267, 4, 384, 896)
    sd = ck.state()
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    print(f"read {len(sd)} tensors")

    # the mesh
    np.random.seed(a.seed)
    v, f = read_obj(a.mesh)
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False, maintain_order=True)
    vn, fn = np.array(m.vertex_normals), np.array(m.face_normals)
    dump("mesh_v", v.astype(np.float32)); dump("mesh_f", f); dump("mesh_vn", vn.astype(np.float32))
    # AugmentAffine (normalize_into [-1, 1]) and AugmentNormalize
    lo, hi = v.min(0), v.max(0)
    scale = float(np.max((hi - lo) / 2.0))
    centre = (hi + lo) / 2
    vnorm = ((v - centre) / scale).astype(np.float32)
    dump("norm", np.concatenate([centre, [scale]]).astype(np.float32))
    dump("mesh_vnorm", vnorm)
    vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-10)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-10)
    sp, sn = sample_mix(vnorm.copy(), f, vn.astype(np.float32), fn.astype(np.float32), 54000, 16384)
    sp, sn = sp.astype(np.float32), sn.astype(np.float32)
    dump("samp_pts", sp); dump("samp_nrm", sn)
    pts, nrm = torch.from_numpy(sp)[None], torch.from_numpy(sn)[None]

    # the shape encoder, output_proj (Linear, then nn.RMSNorm whose eps is bfloat16's machine epsilon)
    ind = np.random.default_rng(seed=0).choice(pts.shape[1], 512 * 4, replace=False)
    lat, sel = mesh_encode(sd, pts, nrm, torch.from_numpy(ind))
    h = lin(lat, sd, "output_proj.0")
    eps = 2.0 ** -7
    cond = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * sd["output_proj.1.weight"]
    dump("enc_ind", ind); dump("enc_fps", sel); dump("enc_lat", lat[0]); dump("cond", cond[0])
    print("encoder done")

    # the skin VAE's condition tokens (its random subset: seeded here, dumped)
    vind = np.random.default_rng(seed=a.seed + 1).choice(pts.shape[1], 384 * 4, replace=False)
    ccat = torch.cat([pts, nrm], -1)
    cl, vsel = vae_cond_encode(sd, ccat, torch.from_numpy(vind))
    dump("vae_ind", vind); dump("vae_fps", vsel); dump("cond_lat", cl[0])
    print("condition tokens done")
    if "tokens" not in stages:
        return

    # the decoder
    from transformers import Qwen3Config, Qwen3ForCausalLM, LogitsProcessor, LogitsProcessorList
    cfg = Qwen3Config(vocab_size=33036, hidden_size=896, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16,
                      num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6, max_position_embeddings=3192, rope_theta=1000000,
                      tie_word_embeddings=False, torch_dtype=torch.float32)
    lm = Qwen3ForCausalLM(cfg).eval()
    tsd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
    miss = lm.load_state_dict(tsd, strict=True)
    print("qwen3 loaded", miss, "tied embeddings equal:", bool(torch.equal(tsd["model.embed_tokens.weight"], tsd["lm_head.weight"])))
    eos_model = 33036 - 1
    start = [tok.bos, tok.cls["articulation"]]
    emb = lm.get_input_embeddings()
    prefix = torch.cat([cond, emb(torch.tensor([start]))], dim=1)

    class Switch(LogitsProcessor):
        """VocabSwitchingLogitsProcessor (src/model/tokenrig.py)"""

        def __init__(self):
            self.init = torch.tensor(start)

        def __call__(self, input_ids, scores):
            for bi, seq in enumerate(input_ids):
                mask = torch.full_like(scores[bi], float("-inf"))
                seq = torch.cat([self.init, seq])
                length = len(seq)
                if tok.eos in seq:
                    mask[tok.eos:] = 0
                    where = torch.where(seq == tok.eos)[0][:1]
                    J = tok.bones_in_sequence(seq.detach().cpu().numpy())
                    if (length - where) == J * 4:
                        mask[:] = float("-inf")
                        mask[eos_model] = 0
                    else:
                        mask[eos_model] = float("-inf")
                else:
                    mask[tok.next_possible(seq.detach().cpu().numpy())] = 0
                scores[bi] = scores[bi] + mask
            return scores

    out = lm.generate(inputs_embeds=prefix, do_sample=False, num_beams=1, repetition_penalty=2.0, max_new_tokens=a.max_new,
                      eos_token_id=eos_model, pad_token_id=tok.pad, bos_token_id=tok.bos, logits_processor=LogitsProcessorList([Switch()]))
    gen = out[0].tolist()
    # transformers may return the prompt's ids too; the embeds path returns the new tokens only
    print("generated", len(gen), "tokens; the last", gen[-3:])
    dump("greedy", np.array(gen, dtype=np.int64))
    full = torch.cat([prefix, emb(torch.tensor([gen]))], dim=1)
    logits = lm(inputs_embeds=full).logits[0, prefix.shape[1] - 1:]
    dump("greedy_logits", logits)
    print("logits", tuple(logits.shape))

    ids = np.array(start + gen)
    where = np.where(ids == tok.eos)[0]
    assert len(where), "no skeleton end"
    w = int(where[0])
    joints, parents = tok.detokenize(ids[:w + 1])
    J = len(joints)
    dump("joints", joints); dump("parents", parents)
    print("skeleton:", J, "joints")
    if "skin" not in stages:
        return
    skin_tokens = ids[w + 1:w + 1 + J * 4]
    if len(skin_tokens) != J * 4:
        print("no skin: the sequence has", len(skin_tokens), "skin tokens, wanted", J * 4)
        return
    idx = torch.from_numpy(skin_tokens - tok.vocab_size).long()
    z = fsq_codes(sd, idx).reshape(J, 4, 512)
    dump("fsq_z", z)
    cond6 = ccat
    cols = []
    for j in range(J):
        col, kv = vae_decode(sd, z[j:j + 1], cl, cond6)
        if j == 0:
            dump("dec_kv0", kv)
        cols.append(col)
        print(f"bone {j + 1}/{J}", flush=True)
    skin = torch.stack(cols, dim=1)
    dump("skin", skin)
    # Asset.from_data: the 8 nearest samples, inverse-distance weights
    tree = cKDTree(sp)
    dist, ix = tree.query(vnorm, k=8)
    wts = 1.0 / (dist + 1e-8)
    wts = wts / wts.sum(axis=1, keepdims=True)
    sk = np.einsum("nk,nkj->nj", wts, skin.numpy()[ix]).astype(np.float32)
    dump("skin_v", sk)
    # the exporter: the 4 largest weights a vertex, renormalized
    order = np.argsort(-sk, axis=1)
    top = np.take_along_axis(sk, order, axis=1)
    keep = min(4, sk.shape[1])
    wt = np.zeros_like(sk)
    for i in range(keep):
        np.put_along_axis(wt, order[:, i:i + 1], top[:, i:i + 1] / top[:, :keep].sum(axis=1, keepdims=True), axis=1)
    dump("weights", wt)
    print("done")


if __name__ == "__main__":
    main()
