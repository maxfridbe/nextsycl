"""P-H3-Image on the CPU in float32 - the reference for nextsycl's engine (image/ph3image): every stage's tensors dumped
for `nextsycl image check`. The pipeline is diffusers' own (huggingface/diffusers 84532a058702c74ae5d1dd6ceb05683fc5d5e768,
src/diffusers/modular_pipelines/p_h3_image): its functions and static methods are called as they are, only the order of
the steps is written out here, so a stage can be dumped and run alone. The weights are PrunaAI/P-H3-Image's (revision
7f98acef), the diffusers layout: the text encoder (Qwen3-VL-4B), the text connector, the VAE, and the transformer - 66 GB
in bfloat16, so its 50 blocks are read one at a time from the shards and widened to float32 for their step (the full
model, its AdaLN branches unfolded; the engine runs ComfyUI's folded table - the `fold` stage measures that difference).
The prompt is rendered as written (no prompt upsampler).

    python -I ref.py --hf HF_DIR --out DIR [--prompt TEXT] [--size 576x384] [--seed 3] [--stages te,dit,vae]
                     [--image PNG ...] [--comfy P-H3-IMAGE.safetensors]

Stages:
  te     tokens and the text encoder (+ the vision tower for an edit's pictures) -> the connector's conditioning
  dit    the packed layout, the noise, the 8 steps over the transformer -> the final latents
  vae    the final latents through the VAE decoder -> the picture
  venc   (an edit) the pictures through the VAE encoder -> the conditioning latents (run before dit)
  fold   ComfyUI's folded AdaLN table (--comfy: a P-H3-Image ComfyUI file) against the full model's tables, per step

Dumps (DIR, .npy float32 unless said; [S] = the packed sequence):
  te:   ids (int32) and tags (int32: 1 text, 0 a picture's vision tokens), prompt.txt; hs_K [T, 2560] for K in
        1, 13, 19, 25, 31, 36 (transformers' hidden_states: K = after layer K, 36 = after the final norm); cond [T, 5120]
        (the connector's output, the sink row first). An edit adds vis_pix_I (the processor's patch rows of picture I),
        vis_grid (int32 [n, 3]), vis_emb_I [n_I, 2560] (the tower's merged tokens) and vis_deep_I_J [n_I, 2560] (its
        deepstack features J = 0, 1, 2), vis_img_I (uint8 [h, w, 3]: picture I as the text encoder's processor got it)
  venc: ref_img_I (uint8 [h, w, 3]: picture I resized for the VAE), ref_lat_I [24, h, w] (the posterior sampled under
        seed 42, rounded to float16, normalized), canvas.txt ("W H")
  dit:  pos [S, 3] (float64), tags_s (int32 [S]), sigmas [9], plan_t_I (the distinct timesteps of step I), plan_i_I
        (int32 [S]); noise_v [rows, 96] and noise_a [4, 32] (the draws, conditioning rows first for an edit);
        lat_I / aud_I (the rows step I starts from), vel_I [rows, 96] / avel_I [4, 32] (the transformer's output); at
        step 0: txt (the token refiner's output [T, 5376]), x_in (the stream before block 0 [S, 5376]), blk_B (after block
        B, for B in 0, 1, 24, 49); lat_final [24, h, w] (unpatchified, normalized)
  vae:  z [24, h, w] (the decoder's input: denormalized), raw [3, H, W] (the decoder's output, ImageNet-normalized),
        rgb [3, H, W] in [0, 1], ref.png
"""
import argparse
import json
import os
import time

import numpy as np
import torch

torch.set_grad_enabled(False)

DUMP_BLOCKS = (0, 1, 24, 49)
TAPS = (1, 13, 19, 25, 31, 36)


def save(out, name, t):
    a = t.detach().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t)
    if a.dtype in (np.float16, np.float64) and name not in ("pos",):
        a = a.astype(np.float32)
    if a.dtype == np.int64:
        a = a.astype(np.int32)
    np.save(os.path.join(out, name + ".npy"), a)
    shown = a.astype(np.float64)
    print(f"  {name} {list(a.shape)} {a.dtype} mean {shown.mean():.5f} std {shown.std():.5f}", flush=True)


def load(out, name):
    return np.load(os.path.join(out, name + ".npy"))


def size(s):
    w, h = (int(v) for v in s.lower().split("x"))
    return w, h


def pictures(a):
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in a.image]


# ---- te -------------------------------------------------------------------------------------------------------------

def stage_te(a):
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from diffusers.models import PH3ImageTextConnector
    from diffusers.modular_pipelines.p_h3_image.before_encoder import _size_at_area
    from diffusers.image_processor import VaeImageProcessor

    tok = Qwen2TokenizerFast.from_pretrained(os.path.join(a.hf, "tokenizer"))
    proc = Qwen3VLProcessor.from_pretrained(os.path.join(a.hf, "processor"))
    vision_inputs = None
    if a.image:
        # PH3ImageEditResizeStep (the text encoder's sizes) and PH3ImageEditTextEncoderStep, as written there
        images = pictures(a)
        resizer = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)
        vis = []
        for im in images:
            sz = _size_at_area(*im.size, 1_000_000, 32)
            vis.append(im if im.size == sz else resizer.resize(im, height=sz[1], width=sz[0]))
        for i, im in enumerate(vis):
            save(a.out, f"vis_img_{i}", np.array(im))
        vision = proc.image_processor(images=vis, return_tensors="pt")
        merge = proc.image_processor.merge_size ** 2
        ids, tags = [], []
        for i, grid in enumerate(vision["image_grid_thw"]):
            label = tok(f"<Picture {i + 1}>: ", add_special_tokens=False)["input_ids"]
            vids = ([tok.convert_tokens_to_ids("<|vision_start|>")] + [tok.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge)
                    + [tok.convert_tokens_to_ids("<|vision_end|>")])
            ids += label + vids
            tags += [1] * len(label) + [0] * len(vids)
        prompt = a.prompt
        if not prompt.lstrip().startswith(("Edit <Picture 1>.", "subject_definitions:")):
            prompt = f"Edit <Picture 1>. {prompt}"
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids += pids
        tags += [1] * len(pids)
        vision_inputs = {"pixel_values": vision["pixel_values"], "image_grid_thw": vision["image_grid_thw"]}
        save(a.out, "vis_grid", vision["image_grid_thw"])
        n0 = 0
        for i, g in enumerate(vision["image_grid_thw"]):
            n = int(g.prod())
            save(a.out, f"vis_pix_{i}", vision["pixel_values"][n0:n0 + n])
            n0 += n
    else:
        prompt = a.prompt
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        tags = [1] * len(ids)
    open(os.path.join(a.out, "prompt.txt"), "w").write(prompt)
    save(a.out, "ids", np.array(ids, np.int32))
    save(a.out, "tags", np.array(tags, np.int32))

    te = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(a.hf, "text_encoder"), dtype=torch.float32).eval()
    con = PH3ImageTextConnector.from_pretrained(os.path.join(a.hf, "text_connector"), torch_dtype=torch.float32).eval()
    seen = {}
    if vision_inputs is not None:
        def grab(_m, _i, o):
            seen["vis"] = o
        te.model.visual.register_forward_hook(grab)
    # get_p_h3_image_prompt_embeds, with the hidden states kept
    input_ids = torch.tensor([ids], dtype=torch.long)
    mm = torch.tensor(proc.create_mm_token_type_ids([ids]), dtype=torch.long)
    kw = {k: v.to(torch.float32) if k.startswith("pixel_") else v for k, v in (vision_inputs or {}).items()}
    t = time.time()
    o = te(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), mm_token_type_ids=mm, use_cache=False, output_hidden_states=True,
           logits_to_keep=1, **kw)
    print(f"te {time.time() - t:.1f}s, {len(o.hidden_states)} hidden states")
    for k in TAPS:
        save(a.out, f"hs_{k}", o.hidden_states[k][0])
    save(a.out, "cond", con(o.hidden_states)[0])
    if "vis" in seen:
        v = seen["vis"]
        # Qwen3VLVisionModel returns (merged tokens, deepstack features) - as a tuple or an output object by version
        emb, deep = (v[0], v[1]) if isinstance(v, (tuple, list)) else (v.pooler_output, v.deepstack_features)
        n0 = 0
        for i, g in enumerate(vision_inputs["image_grid_thw"]):
            n = int(g.prod()) // 4
            save(a.out, f"vis_emb_{i}", emb[n0:n0 + n])
            for j, d in enumerate(deep):
                save(a.out, f"vis_deep_{i}_{j}", d[n0:n0 + n])
            n0 += n


# ---- venc -----------------------------------------------------------------------------------------------------------

def stage_venc(a):
    from diffusers.models import AutoencoderKLPH3Image
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.modular_pipelines.p_h3_image.before_encoder import _size_in_area_range, round_canvas
    from diffusers.modular_pipelines.p_h3_image.encoders import encode_vae_condition

    images = pictures(a)
    multiple = 32
    sizes = [_size_in_area_range(*im.size, 1024 ** 2, 8 * 1024 ** 2, multiple) for im in images]
    if a.canvas:
        w, h = size(a.canvas)
        h, w = round_canvas(h, w, multiple)
    else:
        sizes[0] = _size_in_area_range(*images[0].size, 4 * 1024 ** 2, 8 * 1024 ** 2, multiple)
        w, h = sizes[0]
    open(os.path.join(a.out, "canvas.txt"), "w").write(f"{w} {h}\n")
    resizer = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)
    refs = [im if im.size == sz else resizer.resize(im, height=sz[1], width=sz[0]) for im, sz in zip(images, sizes)]
    vae = AutoencoderKLPH3Image.from_pretrained(os.path.join(a.hf, "vae"), torch_dtype=torch.float32).eval()
    for i, im in enumerate(refs):
        px = np.array(im)
        save(a.out, f"ref_img_{i}", px)
        t = time.time()
        lat = encode_vae_condition(vae, torch.from_numpy(px).permute(2, 0, 1)[None, :, None], (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 42)
        print(f"venc {i}: {time.time() - t:.1f}s")
        save(a.out, f"ref_lat_{i}", lat[0, :, 0])


# ---- dit ------------------------------------------------------------------------------------------------------------

class Streamed:
    """MiniMaxH3Transformer3DModel with its blocks read from the shards one at a time (float32 for their call)"""

    def __init__(self, tdir):
        from safetensors import safe_open
        from diffusers.models import MiniMaxH3Transformer3DModel
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
        cfg = {k: v for k, v in json.load(open(os.path.join(tdir, "config.json"))).items() if not k.startswith("_")}
        with torch.device("meta"):
            self.m = MiniMaxH3Transformer3DModel(**cfg)
        self.index = json.load(open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
        self.files = {f: safe_open(os.path.join(tdir, f), "pt") for f in set(self.index.values())}
        for name, mod in self.m.named_children():
            if name != "transformer_blocks" and any(True for _ in mod.parameters()):
                self.materialize(mod, name + ".")
        self.m.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=cfg.get("rope_freq_dim", 16), rope_theta=cfg.get("rope_theta", 10000.0))

    def materialize(self, mod, prefix):
        sd = {k[len(prefix):]: self.files[f].get_tensor(k).to(torch.float32) for k, f in self.index.items() if k.startswith(prefix)}
        mod.to_empty(device="cpu")
        mod.load_state_dict(sd, strict=True, assign=True)

    def forward(self, taps, hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices, token_tags, position_ids,
                video_indices, audio_indices, text_indices):
        """MiniMaxH3Transformer3DModel.forward, the block loop streaming; `taps(name, tensor)` sees the stages"""
        m = self.m
        rotary_emb = m.rope(position_ids)
        video_embeds = m.proj_in(hidden_states)
        audio_embeds = m.audio_proj_in(audio_hidden_states)
        text_embeds = m.token_refiner(m.context_embedder(encoder_hidden_states))
        taps("txt", text_embeds[0])
        s = position_ids.shape[0]
        x = text_embeds.new_zeros((text_embeds.shape[0], s, text_embeds.shape[-1]))
        x = x.index_copy(1, text_indices, text_embeds)
        x = x.index_copy(1, video_indices, video_embeds)
        x = x.index_copy(1, audio_indices, audio_embeds)
        taps("x_in", x[0])
        temb = m.time_embedder(m.time_proj(timestep))
        adaln_indices = timestep_indices * 3 + token_tags
        for i, blk in enumerate(m.transformer_blocks):
            self.materialize(blk, f"transformer_blocks.{i}.")
            x = blk(x, temb, adaln_indices, rotary_emb)
            blk.to("meta")
            taps(f"blk_{i}", x[0])
        x = m.norm_out(x, temb, timestep_indices)
        return m.proj_out(x).index_select(1, video_indices), m.audio_proj_out(x).index_select(1, audio_indices)


def layout(a, tags, conds):
    """The packed sequence of PH3ImagePrepareLayoutStep / PH3ImageEditPrepareLayoutStep"""
    from diffusers.modular_pipelines.p_h3_image.before_denoise import PH3ImageEditPrepareLayoutStep, PH3ImagePrepareLayoutStep
    from diffusers.modular_pipelines.p_h3_image.before_encoder import round_canvas
    if conds:
        w, h = (int(v) for v in open(os.path.join(a.out, "canvas.txt")).read().split())
        lay = PH3ImageEditPrepareLayoutStep.build_edit_packed_sequence(tags, conds, h // 16, w // 16, 2, (1, 2, 2), 2, 2, 0)
    else:
        w, h = size(a.size)
        h, w = round_canvas(h, w, 32)
        lay = PH3ImagePrepareLayoutStep.build_packed_sequence(tags, 1, h // 16, w // 16, 2, (1, 2, 2), 2, 2, 0)
    return (w, h), lay


def stage_dit(a):
    from diffusers.schedulers import MiniMaxH3Scheduler
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.modular_pipelines.p_h3_image.before_denoise import PH3ImageSetTimestepsStep, patchify_video_latents

    cond = torch.from_numpy(load(a.out, "cond"))[None]
    tags = torch.from_numpy(load(a.out, "tags")).to(torch.long)
    conds = []
    i = 0
    while os.path.exists(os.path.join(a.out, f"ref_lat_{i}.npy")):
        conds.append(torch.from_numpy(load(a.out, f"ref_lat_{i}"))[None, :, None])
        i += 1
    (w, h), (pos, tags_s, vidx, aidx, tidx, ncv, nca) = layout(a, tags, conds)
    lh, lw = h // 16, w // 16
    print(f"canvas {w}x{h}, latent {lw}x{lh}, {pos.shape[0]} rows ({tidx.numel()} text, {ncv} conditioning)")
    save(a.out, "pos", pos)
    save(a.out, "tags_s", tags_s)
    sched, asched = MiniMaxH3Scheduler.from_pretrained(os.path.join(a.hf, "scheduler")), MiniMaxH3Scheduler.from_pretrained(os.path.join(a.hf, "audio_scheduler"))
    sigmas = json.load(open(os.path.join(a.hf, "modular_model_index.json")))["sample_sigmas"]
    sched.set_timesteps(sigmas=sigmas)
    asched.set_timesteps(sigmas=sigmas)
    save(a.out, "sigmas", sched.sigmas)
    # the draws in the pipeline's order: each condition's augmentation noise, then the image, then the audio
    g = torch.Generator().manual_seed(a.seed)
    packed = []
    for c in conds:
        n = randn_tensor(c.shape, generator=g, dtype=torch.float32)
        packed.append(patchify_video_latents(sched.scale_noise(c, torch.tensor(0.999), n), (1, 2, 2)))
    lat = randn_tensor((1, 24, 1, lh, lw), generator=g, dtype=torch.float32)
    lat = patchify_video_latents(lat, (1, 2, 2))
    aud = randn_tensor((4, 32), generator=g, dtype=torch.float32)
    if packed:
        lat = torch.cat([torch.cat(packed), lat])
    save(a.out, "noise_v", lat)
    save(a.out, "noise_a", aud)
    plans = [PH3ImageSetTimestepsStep.build_row_timesteps(vidx, aidx, ncv, nca, tidx.numel(), float(t), float(at), max(float(t), 0.999), 1.0)
             for t, at in zip(sched.timesteps, asched.timesteps)]
    net = Streamed(os.path.join(a.hf, "transformer"))
    for i, (t, at) in enumerate(zip(sched.timesteps, asched.timesteps)):
        ut, ti = plans[i]
        save(a.out, f"plan_t_{i}", ut)
        save(a.out, f"plan_i_{i}", ti)
        save(a.out, f"lat_{i}", lat)
        save(a.out, f"aud_{i}", aud)

        def taps(name, x):
            if i == 0 and (name in ("txt", "x_in") or name in {f"blk_{b}" for b in DUMP_BLOCKS}):
                save(a.out, name, x)
        t0 = time.time()
        v, av = net.forward(taps, lat[None], aud[None], cond, ut, ti, tags_s, pos, vidx, aidx, tidx)
        print(f"step {i}: t {float(t):.6f}, {time.time() - t0:.0f}s", flush=True)
        save(a.out, f"vel_{i}", v[0])
        save(a.out, f"avel_{i}", av[0])
        # PH3ImageLoopSchedulerStep: only the generated rows move
        lat[ncv:] = sched.step(v[0, ncv:].float(), t, lat[ncv:], return_dict=False)[0]
        aud[nca:] = asched.step(av[0, nca:].float(), asched.timesteps[i], aud[nca:], return_dict=False)[0]
    # PH3ImageAfterDenoiseStep
    rows = lat[ncv:].reshape(1, 1, lh // 2, lw // 2, 24, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7)
    save(a.out, "lat_final", rows.reshape(24, lh, lw))


# ---- vae ------------------------------------------------------------------------------------------------------------

def stage_vae(a):
    from PIL import Image
    from diffusers.models import AutoencoderKLPH3Image
    vae = AutoencoderKLPH3Image.from_pretrained(os.path.join(a.hf, "vae"), torch_dtype=torch.float32).eval()
    lat = torch.from_numpy(load(a.out, "lat_final"))[None, :, None]
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    z = lat * std + mean
    save(a.out, "z", z[0, :, 0])
    t = time.time()
    raw = vae.decode(z, return_dict=False)[0][:, :, 0]
    print(f"vae decode {time.time() - t:.1f}s")
    save(a.out, "raw", raw[0])
    rgb = (raw.float() * torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1) + torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)).clamp(0, 1)
    save(a.out, "rgb", rgb[0])
    # VaeImageProcessor.postprocess -> numpy_to_pil: round(x * 255)
    Image.fromarray((rgb[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(os.path.join(a.out, "ref.png"))


# ---- fold -----------------------------------------------------------------------------------------------------------

def stage_fold(a):
    """ComfyUI's folded AdaLN (adaln_t_table [grid, r] read with linear interpolation, then each block's [6 C * 3, r]
    projection) against the full model's (sinusoid -> time MLP -> silu -> each block's [6 C * 3, 2688] projection), at
    the run's timesteps"""
    from safetensors import safe_open
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    tdir = os.path.join(a.hf, "transformer")
    index = json.load(open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    get = lambda k: safe_open(os.path.join(tdir, index[k]), "pt").get_tensor(k).float()
    proj, emb = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0), TimestepEmbedding(256, 5376, out_dim=2688)
    emb.load_state_dict({k[len("time_embedder."):]: get(k) for k in index if k.startswith("time_embedder.")})
    c = safe_open(a.comfy, "pt")
    table = c.get_tensor("adaln_t_table").float()
    ts = sorted({float(v) for i in range(8) for v in np.load(os.path.join(a.out, f"plan_t_{i}.npy"))}) if os.path.exists(os.path.join(a.out, "plan_t_0.npy")) \
        else [0.0, 0.015247, 0.5, 0.999, 1.0]
    t = torch.tensor(ts)
    full_t = torch.nn.functional.silu(emb(proj(t)))
    g = table.shape[0]
    pos = t.clamp(0, 1) * (g - 1)
    i0 = pos.floor().long().clamp(max=g - 2)
    f = (pos - i0)[:, None]
    fold_t = table[i0] + f * (table[i0 + 1] - table[i0])
    for b in ("transformer_blocks.0", "transformer_blocks.25", "transformer_blocks.49", "norm_out"):
        cb = b.replace("transformer_blocks", "blocks").replace("norm_out", "final_layer")
        full = full_t @ get(f"{b}.{'linear' if b == 'norm_out' else 'adaln_proj.linear'}.weight").T + get(f"{b}.{'linear' if b == 'norm_out' else 'adaln_proj.linear'}.bias")
        fold = fold_t @ c.get_tensor(f"{cb}.adaln_proj.linear.weight").float().T + c.get_tensor(f"{cb}.adaln_proj.linear.bias").float()
        for k, tv in enumerate(ts):
            d = fold[k] - full[k]
            print(f"fold {b} t={tv:.6f}: rel {d.norm() / full[k].norm():.3e} max {d.abs().max():.3e} (rms {full[k].pow(2).mean().sqrt():.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default="A German Expressionist oil painting of a Berlin street at night in the manner of Ernst Ludwig Kirchner, "
                                        "with gaunt figures hurrying past a glowing gas lamp.")
    p.add_argument("--size", default="576x384")
    p.add_argument("--canvas", help="an edit's canvas (default: picture 1's size, at least 4 MP)")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--stages", default="te,dit,vae")
    p.add_argument("--image", action="append", default=[], help="an edit's pictures, picture 1 first")
    p.add_argument("--comfy", help="fold: a P-H3-Image ComfyUI file (its adaln_t_table)")
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    for s in a.stages.split(","):
        t = time.time()
        {"te": stage_te, "venc": stage_venc, "dit": stage_dit, "vae": stage_vae, "fold": stage_fold}[s](a)
        print(f"== {s} {time.time() - t:.0f}s", flush=True)


main()
