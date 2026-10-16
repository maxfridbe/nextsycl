"""Hunyuan3D-2mini-turbo reference dumps for `nextsycl model3d check` (the shape: conditioner, DiT, ShapeVAE decoder).

Runs Tencent's own modules (github.com/Tencent-Hunyuan/Hunyuan3D-2, hy3dgen) in float32 on the CPU from the files
nextsycl serves (the DiT file with its DINOv2-giant conditioner, the turbo VAE), the way the official example
(examples/faster_shape_gen_with_flashvdm_mini_turbo.py) drives them: ImageProcessorV2 at 1022 px, the guidance-distilled
DiT (guidance 5.0, no CFG batch), ConsistencyFlowMatchEuler with 5 steps, the latents shaped by the turbo VAE
(`vae.latent_shape`: the token count the example samples), then the decoder on fixed query points and on a coarse
dense grid.

  python ref_mini.py --repo /var/mnt/2TBSSD/nextsycl-ref/h2 --models ~/models/hunyuan3dmini --image demo.png --out DIR

Writes (float32 .npy, batch dimension dropped):
  pixels.npy     [3, 1022, 1022]  the conditioner's input after its own transform (ImageNet mean / std)
  cond.npy       [5330, 1536]     DINOv2-giant's last hidden state (the DiT's context)
  noise.npy      [N, 64]          the starting latents (torch.manual_seed(seed))
  sigmas.npy     [steps + 1]      the scheduler's sigmas (reversed time: 0 = noise)
  x_<i>.npy, v_<i>.npy            each step's input latents and the DiT's output
  d0_img.npy, d0_txt.npy          double block 0's outputs at step 0 (image rows, context rows)
  latents.npy    [N, 64]          after the last step
  vae_tokens.npy [N, 1024]        the ShapeVAE transformer's output (latents / scale_factor first)
  queries.npy    [Q, 3], logits.npy [Q]   the geometry decoder on fixed points in the box
  grid.npy       [R+1]^3          the dense grid's logits at --res (VanillaVolumeDecoder)
  ref.glb                         skimage marching cubes of that grid (Tencent's export: faces flipped)
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
    """The checkpoint's DINOv2 names (transformers 4.x) as the installed transformers names them (5.x: q_proj ... and
    the SwiGLU's weights_in split into gate_proj | up_proj)"""
    if all(k in want for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        n = (k.replace(".attention.attention.query.", ".attention.q_proj.").replace(".attention.attention.key.", ".attention.k_proj.")
             .replace(".attention.attention.value.", ".attention.v_proj.").replace(".attention.output.dense.", ".attention.o_proj."))
        if ".mlp.weights_in." in n:
            g, u = v.chunk(2, dim=0)
            out[n.replace(".weights_in.", ".gate_proj.")] = g
            out[n.replace(".weights_in.", ".up_proj.")] = u
            continue
        out[n.replace(".mlp.weights_out.", ".mlp.down_proj.")] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="a clone of Tencent-Hunyuan/Hunyuan3D-2")
    ap.add_argument("--models", required=True, help="the folder holding hunyuan3d-dit-v2-mini-turbo/ and hunyuan3d-vae-v2-mini-turbo/")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--res", type=int, default=128, help="the dense grid's resolution (cells a side)")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.repo)
    from PIL import Image
    from safetensors.torch import load_file
    from hy3dgen.shapegen.pipelines import instantiate_from_config, export_to_trimesh
    from hy3dgen.shapegen.models.autoencoders.volume_decoders import VanillaVolumeDecoder
    from hy3dgen.shapegen.models.autoencoders.surface_extractors import MCSurfaceExtractor
    from diffusers.utils.torch_utils import randn_tensor

    os.makedirs(a.out, exist_ok=True)

    def save(name, t):
        t = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
        np.save(os.path.join(a.out, name), t.astype(np.float32))

    dit_dir = os.path.join(a.models, "hunyuan3d-dit-v2-mini-turbo")
    vae_dir = os.path.join(a.models, "hunyuan3d-vae-v2-mini-turbo")
    cfg = yaml.safe_load(open(os.path.join(dit_dir, "config.yaml")))
    t0 = time.time()
    sd = load_file(os.path.join(dit_dir, "model.fp16.safetensors"))
    part = lambda p: {k[len(p):]: v.float() for k, v in sd.items() if k.startswith(p)}
    model = instantiate_from_config(cfg["model"]).float().eval()
    print("dit", model.load_state_dict(part("model."), strict=True))
    conditioner = instantiate_from_config(cfg["conditioner"]).float().eval()
    print("conditioner", conditioner.load_state_dict(dino_names(part("conditioner."), conditioner.state_dict()), strict=True))
    vcfg = yaml.safe_load(open(os.path.join(vae_dir, "config.yaml")))
    vae = instantiate_from_config(vcfg).float().eval()
    # the turbo VAE file is the decoder alone (no point encoder: image-to-3D never encodes)
    r = vae.load_state_dict({k: v.float() for k, v in load_file(os.path.join(vae_dir, "model.fp16.safetensors")).items()}, strict=False)
    assert not r.unexpected_keys and all(k.startswith(("encoder.", "pre_kl.")) for k in r.missing_keys), r
    scheduler = instantiate_from_config(cfg["scheduler"])
    processor = instantiate_from_config(cfg["image_processor"])
    print(f"loaded in {time.time() - t0:.1f} s; latent shape {vae.latent_shape}")

    with torch.no_grad():
        img = Image.open(a.image).convert("RGBA")
        inp = processor(img)
        pix = {}
        enc = conditioner.main_image_encoder
        h = enc.model.register_forward_pre_hook(lambda m, args: pix.setdefault("x", args[0].clone()))
        t0 = time.time()
        cond = conditioner(image=inp["image"])["main"]
        h.remove()
        print(f"conditioner {tuple(cond.shape)} in {time.time() - t0:.1f} s")
        save("pixels.npy", pix["x"][0])
        save("cond.npy", cond[0])

        gen = torch.manual_seed(a.seed)
        latents = randn_tensor((1, *vae.latent_shape), generator=gen, device=torch.device("cpu"), dtype=torch.float32)
        save("noise.npy", latents[0])
        sig = np.linspace(0, 1, a.steps)
        scheduler.set_timesteps(sigmas=sig)
        timesteps = scheduler.timesteps
        save("sigmas.npy", scheduler.sigmas_)
        guidance = torch.tensor([a.guidance], dtype=torch.float32)
        d0 = {}

        def hook(m, args, kwargs, out):
            if "img" not in d0:
                d0["img"], d0["txt"] = out[0].clone(), out[1].clone()

        hh = model.double_blocks[0].register_forward_hook(hook, with_kwargs=True)
        for i, t in enumerate(timesteps):
            ts = t.expand(1).float() / scheduler.config.num_train_timesteps
            t0 = time.time()
            v = model(latents, ts, {"main": cond}, guidance=guidance)
            print(f"step {i} t {ts.item():.4f}: {time.time() - t0:.1f} s")
            save(f"x_{i}.npy", latents[0])
            save(f"v_{i}.npy", v[0])
            latents = scheduler.step(v, t, latents).prev_sample
        hh.remove()
        save("d0_img.npy", d0["img"][0])
        save("d0_txt.npy", d0["txt"][0])
        save("latents.npy", latents[0])

        toks = vae(latents / vae.scale_factor)
        save("vae_tokens.npy", toks[0])
        g = torch.Generator().manual_seed(0)
        q = torch.rand(8192, 3, generator=g) * 2.02 - 1.01
        save("queries.npy", q)
        logits = vae.geo_decoder(queries=q[None], latents=toks)
        save("logits.npy", logits[0, :, 0])
        t0 = time.time()
        grid = VanillaVolumeDecoder()(toks, vae.geo_decoder, bounds=1.01, num_chunks=20000, octree_resolution=a.res, enable_pbar=False)
        print(f"grid {a.res} in {time.time() - t0:.1f} s")
        save("grid.npy", grid[0])
        out = MCSurfaceExtractor()(grid, mc_level=0.0, bounds=1.01, octree_resolution=a.res)
        mesh = export_to_trimesh(out)[0]
        mesh.export(os.path.join(a.out, "ref.glb"))
        json.dump({"tokens": int(vae.latent_shape[0]), "scale_factor": float(vae.scale_factor), "res": a.res, "seed": a.seed,
                   "steps": a.steps, "guidance": a.guidance, "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                   "timesteps": [float(x) for x in timesteps]}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
        print("done", len(mesh.vertices), "vertices", len(mesh.faces), "faces")


if __name__ == "__main__":
    main()
