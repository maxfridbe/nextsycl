"""AnimateAnyMesh's reference dumps, stage by stage, for `nextsycl model3d check` (model3d/animateanymesh/src/check.rs).

Runs the official code (github.com/JarrentWu1031/AnimateAnyMesh: networks/, diffusion/, utils/mesh_utils.py) in
float32 on the CPU, on a mesh read from an OBJ instead of Blender, with two stand-ins:
- pytorch3d's `sample_farthest_points` (its CPU algorithm: start at index 0, take the point farthest from those
  taken, the first one on ties; -1 past a cloud's length);
- the SD-1.5 CLIP text encoder built from its config (CLIP ViT-L/14's text tower) and filled from the RF checkpoint,
  which holds it; the tokenizer read from the local copy.

The OBJ is Y up (glTF's frame); Blender's importers turn it Z up ((x, y, z) -> (x, -z, y)), as the model was trained,
so the same is done here.

    python ref.py --repo SRC/AnimateAnyMesh --models ~/models/animateanymesh --mesh humanoid_lo.obj \
        --prompt "a person waving both arms" --out ~/nsref/rig/aam [--seed 666] [--steps 64] [--cfg 3]

Dumps (float32 .npy; indices as float32):
    verts.npy [N, 3]        merged, recentred and scaled vertices (the model's frame)
    faces.npy [F, 3]        merged faces
    embed0.npy [L, 256]     point embedding of every vertex, padding included (L = max(4096, N))
    embed_nb.npy [L, 256]   after the neighbour attention
    fps.npy [K]             the trajectory vertices (farthest points, the first replaced at random as the code does)
    fps_raw.npy [K]         the farthest points before that replacement
    f0.npy [K, 32]          the rest-shape latents, rescaled (what the flow is clamped to)
    tokens.npy [77]         the prompt's CLIP tokens
    clip.npy [77, 768]      CLIP's last hidden state
    noise.npy [K, 64]       the flow's starting noise
    dit_x.npy, dit_t.npy, dit_out.npy   one DiT forward (cond and uncond rows) at t = 0.3 on the noise with f0
    times.npy [steps]       the ODE grid
    traj_S.npy [K, 64]      the ODE state after step S (S = 1, 2, steps/2, steps - 1)
    samples.npy [K, 64]     the result, f0 clamped, rescaled back (the decoder's input)
    out.npy [16, N, 3]      the decoded frames
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch


def fps_cpu(points, lengths=None, K=50, random_start_point=False):
    """pytorch3d.ops.sample_farthest_points, as its CPU kernel computes it (the distances summed in feature order)"""
    assert not random_start_point
    B, P, D = points.shape
    if lengths is None:
        lengths = torch.full((B,), P, dtype=torch.int64)
    K = K if isinstance(K, int) else int(K)
    idx = torch.full((B, K), -1, dtype=torch.int64)
    for b in range(B):
        n = int(lengths[b])
        pts = points[b, :n]
        dists = torch.full((n,), float("inf"), dtype=points.dtype)
        sel = 0
        idx[b, 0] = 0
        for k in range(1, min(K, n)):
            # as pytorch3d's kernel sums: feature by feature, in float32, a product then a sum
            d = torch.zeros(n, dtype=points.dtype)
            for j in range(D):
                diff = pts[:, j] - pts[sel, j]
                d = d + diff * diff
            dists = torch.minimum(dists, d)
            sel = int(torch.argmax(dists))  # the first maximum
            idx[b, k] = sel
    FPS_LOG.append(idx.clone())
    return None, idx


FPS_LOG = []


def install_stand_ins(models):
    p3d = types.ModuleType("pytorch3d")
    ops = types.ModuleType("pytorch3d.ops")
    ops.sample_farthest_points = fps_cpu
    p3d.ops = ops
    sys.modules["pytorch3d"] = p3d
    sys.modules["pytorch3d.ops"] = ops
    # the code turns the math kernel off for CUDA's flash / memory-efficient ones; on the CPU that leaves none
    import contextlib
    torch.backends.cuda.sdp_kernel = lambda *a, **k: contextlib.nullcontext()
    import transformers
    tok_dir = os.path.join(models, "clip", "tokenizer")

    def text_model(*a, **k):
        cfg = transformers.CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                                          num_attention_heads=12, max_position_embeddings=77, hidden_act="quick_gelu",
                                          projection_dim=768, bos_token_id=0, eos_token_id=2, pad_token_id=1)
        return transformers.CLIPTextModel(cfg)

    real_tokenizer = transformers.CLIPTokenizer.from_pretrained

    def tokenizer(*a, **k):
        return real_tokenizer(tok_dir)

    transformers.CLIPTextModel.from_pretrained = staticmethod(text_model)
    transformers.CLIPTokenizer.from_pretrained = staticmethod(tokenizer)


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
    # Blender's frame: Z up
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    return torch.from_numpy(v), torch.tensor(f, dtype=torch.int64)


def save(out, name, x):
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--max-length", type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    sys.path.insert(0, a.repo)
    install_stand_ins(a.models)
    from networks.configs import model_from_config
    from networks.dymeshvae import DyMeshVAE
    import diffusion.rf_diffusion as rfd
    from utils.mesh_utils import get_adjacency_matrix, merge_identical_vertices_with_indices

    full = json.load(open(os.path.join(a.models, "rf_model", "training_config.json")))
    vae_cfg, rf_cfg = full["vae_config"], full["rf_config"]
    f0c, T = vae_cfg["latent_dim"], vae_cfg["T"]
    st = json.load(open(os.path.join(a.models, "dvae_factors", "dvae_f.json")))
    vae = DyMeshVAE(**vae_cfg)
    vae.load_state_dict(torch.load(os.path.join(a.models, "dvae", "dvae_f.pth"), map_location="cpu"))
    vae.eval()
    rf = model_from_config(rf_cfg, "cpu")
    sd = torch.load(os.path.join(a.models, "rf_model", "rf_epoch_f.pth"), map_location="cpu")
    # transformers 5's CLIPTextModel has no text_model wrapper (the checkpoint was saved with 4.x)
    sd = {k.replace("clip_text_model.text_model.", "clip_text_model."): v for k, v in sd.items()}
    r = rf.load_state_dict(sd, strict=False)
    print("rf load: missing", r.missing_keys[:5], "unexpected", r.unexpected_keys[:5])
    assert not [k for k in r.missing_keys if "position_ids" not in k], "the RF checkpoint does not fill the model"
    rf.eval()

    with torch.no_grad():
        verts, faces = read_obj(a.mesh)
        merged_v, merged_f, _ = merge_identical_vertices_with_indices([verts], [faces])
        vertices, faces = merged_v.to(torch.float32), merged_f.to(torch.int64)
        num_traj = max(512, vertices.shape[0] // 8)
        center = (vertices.max(dim=0)[0] + vertices.min(dim=0)[0]) / 2
        vertices = vertices - center
        v_max = vertices.abs().max()
        vertices = vertices / (v_max + 1e-8)
        save(a.out, "verts", vertices)
        save(a.out, "faces", faces)
        N = vertices.shape[0]
        max_length = max(a.max_length, N)
        faces_max = min(int(max_length * 2.5), faces.shape[0])
        vertices = torch.cat([vertices, torch.zeros(max_length - N, 3)], dim=0)
        faces = torch.cat([faces, -1 * torch.ones(faces_max - faces.shape[0], 3).to(torch.int64)], dim=0)
        vertices = vertices[None, None].repeat(1, T, 1, 1)
        faces = faces[None]
        valid_mask = ~(vertices.permute(0, 2, 1, 3).flatten(2, 3) == 0.0).all(dim=-1)
        valid_length = valid_mask.sum(dim=-1)
        adj = get_adjacency_matrix(vertices[:, 0], faces, valid_length)

        torch.manual_seed(a.seed)
        cap = {}

        def keep(name):
            def hook(m, i, o):
                cap.setdefault(name, o[0].clone())   # (a hook returning a value would replace the output)
            return hook

        vae.point_embed.register_forward_hook(keep("embed0"))
        h = vae.neighbor_layers[0].register_forward_hook(keep("nb_res"))
        x_start = vae(vertices, vertices[:, 0], faces=faces, valid_mask=valid_mask, adj_matrix=adj, num_traj=num_traj, just_encode=True)
        h.remove()
        save(a.out, "embed0", cap["embed0"])
        save(a.out, "embed_nb", cap["embed0"] + cap["nb_res"])
        save(a.out, "fps_raw", FPS_LOG[0][0])
        # the indices encode used (after replace_negative_indices): recovered from the gathered rest latents
        x0 = x_start[0, :, :f0c]
        x0n = (x_start[:, :, :f0c] - st["f0_mean"]) / st["f0_std"]
        xtn = (x_start[:, :, f0c:] - st["ft_mean"]) / st["ft_std"]
        x_start = torch.cat([x0n, xtn], dim=-1)
        f0 = x_start[:, :, :f0c]
        save(a.out, "f0", f0[0])

        # the indices actually used: run the encoder's tail again on the logged ones with the first replaced
        # (replace_negative_indices draws torch.rand; the reference's own draw is reproduced by re-seeding)
        torch.manual_seed(a.seed)
        from networks.dymeshvae import replace_negative_indices
        idx = replace_negative_indices(FPS_LOG[0].clone(), valid_length)
        save(a.out, "fps", idx[0])

        tok = rf.tokenizer([a.prompt], padding="max_length", truncation=True, return_tensors="pt").input_ids
        save(a.out, "tokens", tok[0])
        save(a.out, "clip", rf.clip_text_model(tok)[0][0])

        torch.manual_seed(a.seed + 1)
        noise = torch.randn(x_start.shape)
        save(a.out, "noise", noise[0])
        xt = torch.cat([f0, noise[:, :, f0c:]], dim=-1)
        x2 = torch.cat([xt, xt])
        t = torch.full((2,), 0.3)
        save(a.out, "dit_x", x2)
        save(a.out, "dit_t", t)
        save(a.out, "dit_out", rf(x2, t, texts=[a.prompt, ""]))

        traj = {}
        real_odeint = rfd.odeint

        def odeint(fn, y0, t, **kw):
            save(a.out, "times", t)
            tr = real_odeint(fn, y0, t, **kw)
            traj["all"] = tr
            return tr

        rfd.odeint = odeint
        real_randn = torch.randn
        rfd.torch.randn = lambda *s, **k: noise.clone()
        try:
            samples = rfd.rf_sample(model=rf, shape=x_start.shape, steps=a.steps, model_kwargs=dict(texts=[a.prompt]),
                                    guidance_scale=a.cfg, device="cpu", f0=f0)
        finally:
            rfd.torch.randn = real_randn
        for s in sorted({1, 2, a.steps // 2, a.steps - 1}):
            save(a.out, "traj_%d" % s, traj["all"][s][0])
        samples = torch.cat([samples[:, :, :f0c] * st["f0_std"] + st["f0_mean"], samples[:, :, f0c:] * st["ft_std"] + st["ft_mean"]], dim=-1)
        save(a.out, "samples", samples[0])
        out = vae(vertices, vertices[:, 0], samples=samples, faces=faces, valid_mask=valid_mask, adj_matrix=adj, num_traj=num_traj,
                  just_decode=True)
        save(a.out, "out", out[0][:, :N])
        print("N %d, K %d, x0 rms %.4f, out range %.3f..%.3f" % (N, num_traj, float(x0.pow(2).mean().sqrt()), float(out.min()), float(out.max())))


if __name__ == "__main__":
    main()
