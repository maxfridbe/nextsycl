"""Hunyuan3D 2.1 paint (PBR texture) reference dumps for `nextsycl model3d check` (the paint stages).

Runs Tencent's own code (github.com/Tencent-Hunyuan/Hunyuan3D-2.1, hy3dpaint) in float32 on the CPU from the files
nextsycl serves, the way textureGenPipeline drives it, with three changes for a CPU run, none in the arithmetic:
- custom_rasterizer (a CUDA extension) is replaced by a line-for-line numpy port of its CPU path (rasterizer.cpp);
- the hard-coded `.to("cuda")` calls in the pipeline and SelfAttnProcessor2_0 are skipped (patched);
- the VAE encodes take the posterior's mean, not a sample (its deviation is ~1e-7 there: SD's VAE);
- Blender (bpy, imported for the GLB export only) and PyTorch Lightning (the training module) are stubs.

  python ref_paint.py diffuse --repo .../h21 --models ~/models/hunyuan3d21 --mesh shape.glb --image demo.png --out DIR
  python ref_paint.py bake --repo .../h21 --mesh painted-uv.obj --views DIR --out DIR

diffuse writes (float32 .npy):
  normal.npy, position.npy [6, 512, 512, 3]  the views' maps as the model gets them (2048 renders, PIL-resized)
  ref.npy [512, 512, 3]                     the reference picture (white background)
  dino.npy [257, 1536], dino_proj.npy [3, 1028, 1024]
  ref_latents.npy [4, 64, 64], embeds_normal.npy, embeds_position.npy [6, 4, 64, 64]
  cond_<layer>.npy                          the reference UNet's stored states (down_0_0_0, mid_0_0, up_3_2_0)
  noise.npy [12, 4, 64, 64]; out_0.npy [36, 4, 64, 64] (the UNet at step 0); v_<i>.npy (guided); latents_<i>.npy
  images.npy [12, 512, 512, 3]              the decoded views (albedo x 6, then metallic-roughness x 6), in [0, 1]
bake writes texture.npy, trust.npy (the albedo views baked into the mesh's own UVs; no inpainting)
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

MAXINT = 2147483647
ELEVS = [0, 0, 0, 0, 90, -90]
AZIMS = [0, 90, 180, 270, 0, 180]
WEIGHTS = [1, 0.1, 0.5, 0.1, 0.05, 0.05]


def numpy_rasterizer():
    """custom_rasterizer's CPU path (rasterizer.cpp), in numpy float32"""

    def area2(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1])

    def to_px(v, w, h):
        return np.stack([(v[:, 0] / v[:, 3] * np.float32(0.5) + np.float32(0.5)) * np.float32(w - 1) + np.float32(0.5),
                         (np.float32(0.5) + np.float32(0.5) * v[:, 1] / v[:, 3]) * np.float32(h - 1) + np.float32(0.5),
                         v[:, 2] / v[:, 3] * np.float32(0.49999) + np.float32(0.5)], 1).astype(np.float32)

    def rasterize_image(V, F, D, width, height, occ, use_prior):
        V = V.detach().cpu().numpy().astype(np.float32)
        F = F.detach().cpu().numpy().astype(np.int64)
        vp = to_px(V, width, height)
        z = np.full((height, width), np.int64(MAXINT) * np.int64(MAXINT) + (MAXINT - 1), dtype=np.int64)
        for i, f in enumerate(F):
            t = vp[f]
            x0, x1 = int(t[:, 0].min()), int(t[:, 0].max())
            y0, y1 = int(t[:, 1].min()), int(t[:, 1].max())
            xs = np.arange(max(x0, 0), min(x1 + 1, width))
            ys = np.arange(max(y0, 0), min(y1 + 1, height))
            if xs.size == 0 or ys.size == 0:
                continue
            gx, gy = np.meshgrid(xs.astype(np.float32) + np.float32(0.5), ys.astype(np.float32) + np.float32(0.5))
            a, b, c = t[0], t[1], t[2]
            ar = area2(a, b, c)
            if ar == 0:
                continue
            p = (gx, gy)
            beta = area2(a, p, c) * np.float32(1.0 / ar)
            gamma = area2(a, b, p) * np.float32(1.0 / ar)
            alpha = np.float32(1.0) - beta - gamma
            m = (alpha >= 0) & (alpha <= 1) & (beta >= 0) & (beta <= 1) & (gamma >= 0) & (gamma <= 1)
            if not m.any():
                continue
            depth = alpha * a[2] + beta * b[2] + gamma * c[2]
            tok = (depth * np.float32(2 << 17)).astype(np.int32).astype(np.int64) * MAXINT + (i + 1)
            sub = z[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
            sub[m] = np.minimum(sub[m], tok[m])
        f = z % MAXINT
        bg = f == MAXINT - 1
        findices = np.where(bg, 0, f).astype(np.int32)
        bary = np.zeros((height, width, 3), np.float32)
        idx = np.nonzero(~bg)
        fi = f[idx] - 1
        tri = F[fi]
        gx = idx[1].astype(np.float32) + np.float32(0.5)
        gy = idx[0].astype(np.float32) + np.float32(0.5)
        v0, v1, v2 = vp[tri[:, 0]], vp[tri[:, 1]], vp[tri[:, 2]]
        a2 = lambda a, b, c: (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]) - (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        pp = np.stack([gx, gy], 1)
        ar = a2(v0, v1, v2)
        beta = a2(v0, pp, v2) / ar
        gamma = a2(v0, v1, pp) / ar
        alpha = 1 - beta - gamma
        w = [V[tri[:, k], 3] for k in range(3)]
        b = np.stack([alpha / w[0], beta / w[1], gamma / w[2]], 1)
        b = b / b.sum(1, keepdims=True)
        bary[idx] = b
        return torch.from_numpy(findices), torch.from_numpy(bary)

    def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
        return rasterize_image(pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior)

    def interpolate(col, findices, barycentric, tri):
        f = findices - 1 + (findices == 0)
        vcol = col[0, tri.long()[f.long()]]
        result = barycentric.view(*barycentric.shape, 1) * vcol
        result = torch.sum(result, axis=-2)
        return result.view(1, *result.shape)

    mod = types.ModuleType("custom_rasterizer")
    mod.rasterize = rasterize
    mod.interpolate = interpolate
    return mod


def save(out, name, t):
    t = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
    np.save(os.path.join(out, name), t.astype(np.float32))


def renderer(repo, size, texture_size):
    sys.modules["custom_rasterizer"] = numpy_rasterizer()
    # mesh_utils imports Blender's bpy for its OBJ -> GLB conversion, which is not used here
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))
    sys.path.insert(0, os.path.join(repo, "hy3dpaint"))
    from DifferentiableRenderer.MeshRender import MeshRender
    return MeshRender(default_resolution=size, texture_size=texture_size, bake_mode="back_sample", raster_mode="cr", device="cpu")


def diffuse(a):
    from PIL import Image
    import trimesh
    torch.set_num_threads(a.threads)
    render = renderer(a.repo, 2048, 1024)
    mesh = trimesh.load(a.mesh, force="mesh", process=False)
    # the geometry alone (load_mesh would bake an atlas the mesh does not have yet)
    render.set_mesh(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32))
    t0 = time.time()
    normals = [render.render_normal(e, z, use_abs_coor=True, return_type="pl") for e, z in zip(ELEVS, AZIMS)]
    positions = [render.render_position(e, z, return_type="pl") for e, z in zip(ELEVS, AZIMS)]
    print(f"maps rendered in {time.time() - t0:.1f} s")
    normals = [n.resize((512, 512)) for n in normals]
    positions = [p.resize((512, 512)) for p in positions]
    save(a.out, "normal.npy", np.stack([np.asarray(n, np.float32) / 255 for n in normals]))
    save(a.out, "position.npy", np.stack([np.asarray(p, np.float32) / 255 for p in positions]))
    normals[0].save(os.path.join(a.out, "normal0.png"))
    positions[0].save(os.path.join(a.out, "position0.png"))

    image = Image.open(a.image).resize((512, 512))
    if image.mode == "RGBA":
        white = Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.getchannel("A"))
        image = white
    image = image.convert("RGB")
    save(a.out, "ref.npy", np.asarray(image, np.float32) / 255)

    sys.path.insert(0, os.path.join(a.repo, "hy3dpaint"))
    # the package imports its training module (PyTorch Lightning), which inference never touches
    pl = types.ModuleType("pytorch_lightning")
    pl.LightningModule = torch.nn.Module
    sys.modules.setdefault("pytorch_lightning", pl)
    from diffusers import AutoencoderKL, UniPCMultistepScheduler, DDIMScheduler
    from hunyuanpaintpbr.pipeline import HunyuanPaintPipeline
    from hunyuanpaintpbr.unet.modules import UNet2p5DConditionModel, Dino_v2
    from hunyuanpaintpbr.unet import attn_processor as ap
    from einops import rearrange

    def mda_call(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        B = hidden_states.size(0)
        results = []
        for token, pbr_hs in zip(self.pbr_setting, torch.split(hidden_states, 1, dim=1)):
            hs = rearrange(pbr_hs, "b n_pbrs n l c -> (b n_pbrs n) l c")
            results.append(self.process_single(attn, hs, None, attention_mask, temb, token, False))
        return torch.cat([rearrange(r, "(b n_pbrs n) l c -> b n_pbrs n l c", b=B, n_pbrs=1) for r in results], dim=1)

    ap.SelfAttnProcessor2_0.__call__ = mda_call
    paint = os.path.join(a.models, "hunyuan3d-paintpbr-v2-1")
    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(os.path.join(paint, "vae"), torch_dtype=torch.float32).eval()
    unet = UNet2p5DConditionModel.from_pretrained(os.path.join(paint, "unet"), torch_dtype=torch.float32).eval()
    sched = UniPCMultistepScheduler.from_config(DDIMScheduler.from_pretrained(os.path.join(paint, "scheduler")).config, timestep_spacing="trailing")
    pipe = HunyuanPaintPipeline(vae=vae, text_encoder=None, tokenizer=None, unet=unet, scheduler=sched, feature_extractor=None)
    pipe.set_progress_bar_config(disable=True)
    dino = Dino_v2(os.path.join(a.models, "dinov2-giant")).float().eval()
    print(f"paint models loaded in {time.time() - t0:.1f} s")

    def encode_mean(images):
        B = images.shape[0]
        images = rearrange(images, "b n c h w -> (b n) c h w")
        latents = vae.encode(((images - 0.5) * 2.0).float()).latent_dist.mode() * vae.config.scaling_factor
        return rearrange(latents, "(b n) c h w -> b n c h w", b=B)

    pipe.encode_images = encode_mean
    rec = {"unet": [], "steps": []}
    inner = pipe.unet.forward

    def unet_fwd(*args, **kw):
        out = inner(*args, **kw)
        rec["unet"].append(out[0].detach().clone())
        cache = kw.get("cache") or {}
        if "dino_hidden_states_proj" in cache and "dino_proj" not in rec:
            rec["dino_proj"] = cache["dino_hidden_states_proj"].clone()
            for name in ["down_0_0_0", "mid_0_0", "up_3_2_0"]:
                rec["cond_" + name] = cache["condition_embed_dict"][name][0].clone()
            rec["ref_latents"] = kw["ref_latents"][0, 0].clone()
            rec["embeds_normal"] = kw["embeds_normal"][0].clone()
            rec["embeds_position"] = kw["embeds_position"][0].clone()
        return out

    pipe.unet.forward = unet_fwd
    step = sched.step

    def step_rec(model_output, t, sample, **kw):
        rec["steps"].append((model_output.detach().clone(), sample.detach().clone()))
        out = step(model_output, t, sample, **kw)
        rec["steps"][-1] = rec["steps"][-1] + (out[0].detach().clone() if isinstance(out, tuple) else out.prev_sample.detach().clone(),)
        return out

    sched.step = step_rec
    with torch.no_grad():
        dino_states = dino(image)
        save(a.out, "dino.npy", dino_states[0])
        to_t = lambda ims: torch.from_numpy(np.stack([np.asarray(i, np.float32) / 255 for i in ims])).permute(0, 3, 1, 2)[None]
        noise = torch.randn((12, 4, 64, 64), generator=torch.Generator().manual_seed(0))
        save(a.out, "noise.npy", noise)
        t0 = time.time()
        out = pipe(
            [image], prompt="high quality", num_inference_steps=a.steps, guidance_scale=3.0, width=512, height=512,
            num_in_batch=6, images_normal=to_t(normals), images_position=to_t(positions), dino_hidden_states=dino_states,
            latents=noise, output_type="np",
        )
        print(f"diffusion in {time.time() - t0:.1f} s")
    save(a.out, "images.npy", np.asarray(out.images, np.float32))
    for k in ["dino_proj", "ref_latents", "embeds_normal", "embeds_position", "cond_down_0_0_0", "cond_mid_0_0", "cond_up_3_2_0"]:
        save(a.out, k + ".npy", rec[k])
    save(a.out, "out_0.npy", rec["unet"][0])
    for i, (v, x, nx) in enumerate(rec["steps"]):
        save(a.out, f"v_{i}.npy", v)
        save(a.out, f"x_{i}.npy", x)
        save(a.out, f"latents_{i}.npy", nx)
    json.dump({"steps": a.steps, "timesteps": [int(t) for t in sched.timesteps], "sigmas": [float(s) for s in sched.sigmas]},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    for i, im in enumerate(out.images):
        Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(os.path.join(a.out, f"view{i}.png"))
    print("done")


def bake(a):
    import trimesh
    torch.set_num_threads(a.threads)
    render = renderer(a.repo, 2048, a.texture)
    mesh = trimesh.load(a.mesh, force="mesh", process=False)
    render.load_mesh(mesh=mesh)
    from PIL import Image
    views = [Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8)).resize((2048, 2048)) for v in np.load(os.path.join(a.views, "images.npy"))[:6]]
    tex, cos = [], []
    for v, e, z, w in zip(views, ELEVS, AZIMS, WEIGHTS):
        t, c, _ = render.back_project(v, e, z)
        tex.append(t)
        cos.append(w * c ** 4)
    texture, trust = render.fast_bake_texture(tex, cos)
    save(a.out, "texture.npy", texture)
    save(a.out, "trust.npy", trust.float())
    print("baked", float(trust.float().mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["diffuse", "bake"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--models", default=os.path.expanduser("~/models/hunyuan3d21"))
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--image")
    ap.add_argument("--views")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--texture", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    diffuse(a) if a.what == "diffuse" else bake(a)


if __name__ == "__main__":
    main()
