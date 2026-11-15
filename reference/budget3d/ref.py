"""reference/budget3d/ref.py: Budget3D-Idol (and -Turbo) on the CPU through its own modules (BudgetAny/Budget3D and
speridlabs/idol-sdk at fetch.sh's commits), on the SAME files nextsycl reads, dumping every stage the engine is checked
against (`nextsycl model3d check DIR --model budget3d-idol ...`). float32 throughout, as Budget3D runs off CUDA (its bf16
autocast is CUDA-only); Idol's decoders keep their own half blocks.

    sh fetch.sh CODE
    python -I ref.py OUT --code CODE --idol IDOL_DIR --ckpt budget3d_idol.safetensors [--turbo] [--seed 42]
                     [--budgets 1,0.5,0.25] [--steps 6] [--sample-budget 0.5] [--stages dit,sample,dec] [--gaussians 65536]

IDOL_DIR holds speridlabs/idol's config.json, denoiser.safetensors (the Sobol positions, pos_pe), decoder.safetensors and
decoder_gs.safetensors. The image encoders (DINOv3 ViT-H+/16, the FLUX.2 VAE) are gated: the condition here is a seeded
normal draw of their shapes (feature1 [4101, 1280], feature2 [4101, 128] with its 5 leading rows zero, as Idol pads it),
which is what Budget3D's own parity gate (models/load.py check_dense_parity) feeds.

Dumps (float32 .npy, the batch dimension dropped; indices as float32):
  cond_f1, cond_f2                     the condition
  z_noise [8192,16], cam_noise [1,5]   torch.manual_seed(seed) draws, as the sampler makes them
  ctx_pos, ctx_neg [4101,1024]         the refined condition (the zero condition's for the unguided branch); turbo: ctx_slow,
                                       ctx_fast (the context refiner carries the phase's LoRA)
  dit  per budget B (percent) and call k (t = 1.0, 0.6, 0.3):
       bB_s4 [512], bB_s2 [4096]       detail scores of the noise; bB_idx4, bB_idx2, bB_idx1 the layout; bB_tok [N,1024]
       bB_kK_{nref, joint, head, vtok, vpts, v, vcam, blk0, blk23}   the noise refiner's output, the joint blocks' token rows
                                       (base: [2, N, 1024], the guided and unguided branches), the head (h_tok), out_layer's
                                       v_tok, the expanded per-point velocity (octree order [*, 4096, 2, 16]), the final
                                       velocity [8192,16] (base: guided), the camera's, the first and last joint blocks' outputs
  sample  sample_z<i> [8192,16] / sample_cam<i> after each step, sample_layout<i>_{idx4,idx2,idx1} when built, sample_z /
          sample_cam the result
  dec   from sample_z: dec_l<l>_{x, logits} (each octree level's parent cells and their 8 logits), dec_anchors [K,3],
        gs_feat [K,480], gs_{xyz, fdc, scaling, rotation, opacity} (activated Gaussians)
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--code", required=True, help="fetch.sh's directory (idol-sdk/ and budget3d/)")
    ap.add_argument("--idol", required=True, help="speridlabs/idol's files")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--turbo", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond-seed", type=int, default=0)
    ap.add_argument("--budgets", default="1,0.5,0.25")
    ap.add_argument("--steps", type=int, default=0, help="the sample stage's steps (default 6, turbo 5)")
    ap.add_argument("--sample-budget", type=float, default=0.5)
    ap.add_argument("--stages", default="dit,sample,dec")
    ap.add_argument("--gaussians", type=int, default=65536)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(HERE, "stubs"))
    sys.path.insert(0, os.path.join(a.code, "budget3d"))
    os.environ["IDOL_ROOT"] = os.path.join(a.code, "idol-sdk")

    import numpy as np
    import torch
    from tensordict import TensorDict
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count() or 8)))

    from budget3d.models.load import load_budget3d, load_decoders, load_student, load_trained
    from budget3d.octree import N_SOBOL, build_layout, detail_scores, snap_realizable_tokens
    from budget3d.sampling.sampler import time_grid
    from idol.models import OctreeProbabilityFixedlenDecoder

    os.makedirs(a.out, exist_ok=True)
    stages = set(a.stages.split(","))

    def save(name, t):
        t = t.detach().float().cpu()
        np.save(os.path.join(a.out, name + ".npy"), t.numpy().astype(np.float32))

    def load(name):
        return torch.from_numpy(np.load(os.path.join(a.out, name + ".npy")))

    t0 = time.time()
    if a.turbo:
        from budget3d.models.turbo import PhaseState, bind_phase, inject
        model = load_student(a.idol)
        state = PhaseState(True)
        inject(model, state)
        bind_phase(model, state)
        load_trained(model, a.ckpt)
        loras = [n for n, _ in model.named_parameters() if n.rsplit(".", 1)[-1] in ("slow_A", "slow_B", "fast_A", "fast_B")]
        assert loras, "no LoRA parameters injected"
        model = model.bind_geometry().eval()
    else:
        model, _ = load_budget3d(a.ckpt, device="cpu", weights_dir=a.idol)
    print(f"[ref] loaded {a.ckpt} in {time.time() - t0:.1f} s", flush=True)

    g = torch.Generator().manual_seed(a.cond_seed)
    f1 = torch.randn(1, 4101, 1280, generator=g)
    f2 = torch.randn(1, 4101, 128, generator=g)
    f2[:, :5] = 0
    cond = TensorDict({"feature1": f1, "feature2": f2}, batch_size=1)
    save("cond_f1", f1[0])
    save("cond_f2", f2[0])
    torch.manual_seed(a.seed)
    z = torch.randn(1, model.q_token_length, model.in_channels)
    cam = torch.randn(1, 1, model.cam_channels)
    save("z_noise", z[0])
    save("cam_noise", cam[0])

    if a.turbo:
        from budget3d.models.turbo import PHASE_THRESHOLD
        from budget3d.sampling.turbo import embed_context_phase
        ctx = {True: embed_context_phase(model, cond, True), False: embed_context_phase(model, cond, False)}
        save("ctx_slow", ctx[True][0])
        save("ctx_fast", ctx[False][0])
    else:
        h_pos = model.embed_context(cond)
        h_neg = model.embed_context(cond.apply(torch.zeros_like))
        save("ctx_pos", h_pos[0])
        save("ctx_neg", h_neg[0])
    print(f"[ref] condition refined ({time.time() - t0:.1f} s)", flush=True)

    # what a call computes on the way, recorded by wrapping the model's own pieces
    rec = {}

    def wrap(name):
        orig = getattr(model, name)

        def f(*args, **kw):
            out = orig(*args, **kw)
            rec.setdefault(name, out)
            return out
        setattr(model, name, f)
    for n in ("refine_latent_and_camera", "joint", "head", "_expand"):
        wrap(n)
    hooks = {}
    for tag, mod in (("blk0", model.blocks[0]), ("blk23", model.blocks[-1])):
        mod.register_forward_hook(lambda m, i, o, tag=tag: hooks.setdefault(tag, o))

    def layout_of(ref, n_tokens):
        s4, s2 = detail_scores(ref[:, model.perm_to_octree].reshape(1, 16, 16, 16, 2, -1))
        return s4, s2, build_layout(s4, s2, n_tokens)

    def call(zz, cc, t, layout):
        """One velocity call as the sampler makes it: (v [1, 8192, 16], v_cam [1, 1, 5])"""
        tt = torch.full((1,), 1000 * t)
        if a.turbo:
            out = model(TensorDict({"latent": zz, "camera": cc}, batch_size=1), tt, h_cond=ctx[t >= PHASE_THRESHOLD], layout=layout)
            return out["latent"], out["camera"]
        return model.forward_cfg(zz, cc, tt, h_pos, h_neg, layout, cfg=3.0)

    if "dit" in stages:
        for b in [float(x) for x in a.budgets.split(",")]:
            tag = f"b{round(b * 100)}"
            n = snap_realizable_tokens(b * N_SOBOL)
            s4, s2, layout = layout_of(z, n)
            save(f"{tag}_s4", s4[0])
            save(f"{tag}_s2", s2[0])
            save(f"{tag}_idx4", layout.idx4[0])
            save(f"{tag}_idx2", layout.idx2[0])
            save(f"{tag}_idx1", layout.idx1[0])
            save(f"{tag}_tok", model.embed_tokens(z, layout)[0])
            for k, t in enumerate((1.0, 0.6, 0.3)):
                rec.clear()
                hooks.clear()
                v, vc = call(z, cam, t, layout)
                save(f"{tag}_k{k}_nref", rec["refine_latent_and_camera"][0])
                save(f"{tag}_k{k}_joint", rec["joint"][0])
                save(f"{tag}_k{k}_head", rec["head"][0])
                save(f"{tag}_k{k}_vtok", model.out_layer(rec["head"][0]))
                if "_expand" in rec:
                    save(f"{tag}_k{k}_vpts", rec["_expand"][0])
                save(f"{tag}_k{k}_blk0", hooks["blk0"])
                save(f"{tag}_k{k}_blk23", hooks["blk23"])
                save(f"{tag}_k{k}_v", v[0])
                save(f"{tag}_k{k}_vcam", vc[0])
                print(f"[ref] {tag} {layout} call {k} (t {t}) ({time.time() - t0:.1f} s)", flush=True)

    if "sample" in stages:
        # the samplers' loops (budget3d/sampling/sampler.py sample, turbo.py sample_turbo), with every step dumped
        steps = a.steps or (5 if a.turbo else 6)
        n = snap_realizable_tokens(a.sample_budget * N_SOBOL)
        grid = time_grid(steps, 3.0)
        zz, cc, z0, layout = z.clone(), cam.clone(), z, None
        for i in range(steps):
            t, t_prev = float(grid[i]), float(grid[i + 1])
            build = layout is None or (a.turbo or (i <= 5 and i % 5 == 0))
            if build:
                _, _, layout = layout_of(zz if i == 0 else z0, n)
                save(f"sample_layout{i}_idx4", layout.idx4[0])
                save(f"sample_layout{i}_idx2", layout.idx2[0])
                save(f"sample_layout{i}_idx1", layout.idx1[0])
            v, vc = call(zz, cc, t, layout)
            z0 = zz - t * v
            zz = zz - (t - t_prev) * v
            cc = cc - (t - t_prev) * vc
            save(f"sample_z{i}", zz[0])
            save(f"sample_cam{i}", cc[0])
            print(f"[ref] step {i} t {t:.4f} {layout} ({time.time() - t0:.1f} s)", flush=True)
        save("sample_z", zz[0])
        save("sample_cam", cc[0])
        json.dump({"steps": steps, "budget": a.sample_budget, "tokens": n, "turbo": a.turbo, "seed": a.seed},
                  open(os.path.join(a.out, "sample.json"), "w"))

    if "dec" in stages:
        dec, dec_gs, cfg = load_decoders(a.idol)
        latent = load("sample_z")[None]
        levels = []
        h = dec.register_forward_hook(lambda m, i, o: levels.append((i[0][0].clone(), int(i[1][0]), o[0].clone())))
        torch.manual_seed(a.seed)
        per = dec_gs.rep_config["num_gaussians"]
        anchors = OctreeProbabilityFixedlenDecoder.sample(dec, latent, num_points=max(1, a.gaussians // per),
                                                          level=cfg["max_voxel_level"])
        h.remove()
        for l, (x, res, logits) in enumerate(levels, 1):
            save(f"dec_l{l}_x", x)
            save(f"dec_l{l}_logits", logits)
        save("dec_anchors", anchors[0])
        feats = dec_gs(anchors, latent)
        save("gs_feat", feats[0])
        gs = dec_gs.to_representation(anchors, feats)[0]
        save("gs_xyz", gs.xyz)
        save("gs_fdc", gs.features_dc.reshape(-1, 3))
        save("gs_scaling", gs.scaling)
        save("gs_rotation", gs.rotation)
        save("gs_opacity", gs.opacity.reshape(-1))
        print(f"[ref] decoded {anchors.shape[1]} anchors, {gs.xyz.shape[0]} Gaussians ({time.time() - t0:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
