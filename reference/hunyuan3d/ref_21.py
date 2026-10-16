"""Hunyuan3D 2.1 reference dumps for `nextsycl model3d check` (the shape: conditioner, DiT, ShapeVAE decoder).

Runs Tencent's own modules (github.com/Tencent-Hunyuan/Hunyuan3D-2.1, hy3dshape) in float32 on the CPU from the
file nextsycl serves (Comfy-Org's repackage: the same tensors under model. / vae. / conditioner.), the way
Hunyuan3DDiTFlowMatchingPipeline drives them: ImageProcessorV2 at 512 px, DINOv2-Large at 518, CFG (the conditional
pass, then the unconditional one with an all-zero context), FlowMatchEuler over linspace(0, 1, steps) and a final 1.
The CPU is slow at 3 billion parameters: a few steps (--steps 5) make the sampler's dump; the per-step parity does not
depend on the count.

  python ref_21.py --repo /var/mnt/2TBSSD/nextsycl-ref/h21 --model ~/models/hunyuan3d21 --image demo.png --out DIR

Writes (float32 .npy, batch dimension dropped):
  pixels.npy [3, 518, 518], cond.npy [1370, 1024], noise.npy [4096, 64], sigmas.npy [steps + 1]
  x_<i>.npy, v_<i>.npy      each step's input latents and the guided velocity
  vc_0.npy, vu_0.npy        step 0's conditional and unconditional passes; b0.npy the stream after block 0 (cond)
  latents.npy, vae_tokens.npy [4096, 1024], queries.npy [Q, 3], logits.npy [Q], grid.npy [R+1]^3, ref.glb
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml


def dino_names(sd, want):
    """The checkpoint's DINOv2 names (transformers 4.x) as the installed transformers names them (5.x: q_proj ...)"""
    if all(k in want for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        n = (k.replace(".attention.attention.query.", ".attention.q_proj.").replace(".attention.attention.key.", ".attention.k_proj.")
             .replace(".attention.attention.value.", ".attention.v_proj.").replace(".attention.output.dense.", ".attention.o_proj."))
        out[n] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="a clone of Tencent-Hunyuan/Hunyuan3D-2.1")
    ap.add_argument("--model", required=True, help="the folder holding hunyuan_3d_v2.1.safetensors and hunyuan3d-dit-v2-1/config.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--res", type=int, default=48, help="the dense grid's resolution (cells a side; the CPU is slow)")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.join(a.repo, "hy3dshape"))
    from PIL import Image
    from safetensors.torch import load_file
    from hy3dshape.pipelines import instantiate_from_config, export_to_trimesh
    from hy3dshape.models.autoencoders.volume_decoders import VanillaVolumeDecoder
    from hy3dshape.models.autoencoders.surface_extractors import MCSurfaceExtractor
    from diffusers.utils.torch_utils import randn_tensor

    os.makedirs(a.out, exist_ok=True)

    def save(name, t):
        t = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
        np.save(os.path.join(a.out, name), t.astype(np.float32))

    cfg = yaml.safe_load(open(os.path.join(a.model, "hunyuan3d-dit-v2-1", "config.yaml")))
    t0 = time.time()
    sd = load_file(os.path.join(a.model, "hunyuan_3d_v2.1.safetensors"))
    part = lambda p: {k[len(p):]: v.float() for k, v in sd.items() if k.startswith(p)}
    model = instantiate_from_config(cfg["model"]).float().eval()
    print("dit", model.load_state_dict(part("model."), strict=True))
    conditioner = instantiate_from_config(cfg["conditioner"]).float().eval()
    print("conditioner", conditioner.load_state_dict(dino_names(part("conditioner."), conditioner.state_dict()), strict=True))
    vae = instantiate_from_config(cfg["vae"]).float().eval()
    print("vae", vae.load_state_dict(part("vae."), strict=True))
    del sd
    scheduler = instantiate_from_config(cfg["scheduler"])
    processor = instantiate_from_config(cfg["image_processor"])
    print(f"loaded in {time.time() - t0:.1f} s; latent shape {vae.latent_shape}")

    with torch.no_grad():
        img = Image.open(a.image).convert("RGBA")
        inp = processor(img)
        pix = {}
        h = conditioner.main_image_encoder.model.register_forward_pre_hook(lambda m, args: pix.setdefault("x", args[0].clone()))
        cond = conditioner(image=inp["image"])["main"]
        h.remove()
        uncond = conditioner.unconditional_embedding(1)["main"]
        save("pixels.npy", pix["x"][0])
        save("cond.npy", cond[0])

        latents = randn_tensor((1, *vae.latent_shape), generator=torch.manual_seed(a.seed), device=torch.device("cpu"), dtype=torch.float32)
        save("noise.npy", latents[0])
        scheduler.set_timesteps(sigmas=np.linspace(0, 1, a.steps))
        save("sigmas.npy", scheduler.sigmas)
        ctx = {"main": torch.cat([cond, uncond])}
        b0 = {}
        def hook(m, args, out):
            # (returning nothing: a forward hook's return value would replace the block's output)
            if "x" not in b0:
                b0["x"] = out[0].clone()

        hh = model.blocks[0].register_forward_hook(hook)
        for i, t in enumerate(scheduler.timesteps):
            ts = (t.expand(2).float() / scheduler.config.num_train_timesteps)
            t0 = time.time()
            out = model(torch.cat([latents] * 2), ts, ctx)
            vc, vu = out.chunk(2)
            v = vu + a.guidance * (vc - vu)
            print(f"step {i} t {ts[0].item():.4f}: {time.time() - t0:.1f} s")
            save(f"x_{i}.npy", latents[0])
            save(f"v_{i}.npy", v[0])
            if i == 0:
                save("vc_0.npy", vc[0])
                save("vu_0.npy", vu[0])
                save("b0.npy", b0["x"])
            latents = scheduler.step(v, t, latents).prev_sample
        hh.remove()
        save("latents.npy", latents[0])

        toks = vae(latents / vae.scale_factor)
        save("vae_tokens.npy", toks[0])
        q = torch.rand(8192, 3, generator=torch.Generator().manual_seed(0)) * 2.02 - 1.01
        save("queries.npy", q)
        save("logits.npy", vae.geo_decoder(queries=q[None], latents=toks)[0, :, 0])
        t0 = time.time()
        grid = VanillaVolumeDecoder()(toks, vae.geo_decoder, bounds=1.01, num_chunks=8000, octree_resolution=a.res, enable_pbar=False)
        print(f"grid {a.res} in {time.time() - t0:.1f} s")
        save("grid.npy", grid[0])
        out = MCSurfaceExtractor()(grid, mc_level=0.0, bounds=1.01, octree_resolution=a.res)
        mesh = export_to_trimesh(out)[0]
        mesh.export(os.path.join(a.out, "ref.glb"))
        json.dump({"tokens": int(vae.latent_shape[0]), "scale_factor": float(vae.scale_factor), "res": a.res, "seed": a.seed,
                   "steps": a.steps, "guidance": a.guidance, "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                   "timesteps": [float(x) for x in scheduler.timesteps]}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
        print("done", len(mesh.vertices), "vertices", len(mesh.faces), "faces")


if __name__ == "__main__":
    main()
