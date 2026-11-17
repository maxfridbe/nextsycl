"""MPEG-Diffusion's two-track pieces as scrya-com/mpeg-diffusion computes them (lot/two_track_h3.py, lot/mixed_res.py,
its vendored musubi-tuner H3), against the engine's (crates/h3/src/mpeg.rs). Their code is driven, not copied: the
upstream checkout is imported from --upstream. CPU, float32. docs/ports/mpeg-diffusion.md has the numbers.

    # fixtures for the Rust tests (crates/h3/tests/data/mpeg_fixture.json): the plan, the packed positions, the region
    # mask from scores, the Level-of-Token layout, the tail's sigmas - upstream's own functions on fixed inputs
    python3 -I tt_ref.py fixtures --upstream <mpeg-diffusion checkout> --out mpeg_fixture.json

    # one two-track step of the engine's dump (a generate job with "mpeg_dump": the first two-track call's inputs and
    # outputs, block 0's raw qkv and attention output, the region head's input and output): the region head and mask, the
    # K/V map, block 0's dual-resolution attention, the layout and positions, and the whole network call through
    # upstream's two_track_forward (the int8 checkpoint dequantized to float32, one block in memory at a time)
    python3 -I tt_ref.py check --upstream <checkout> --int8 minimax_h3_fl2va_pruned_int8_convrot.safetensors \
        --maps up_maps_tt.safetensors --head region_head_tt.safetensors --dump dump.safetensors [--blocks N] [--no-step]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

H = None


def setup(upstream):
    up = Path(upstream)
    sys.path.insert(0, str(up / "third_party"))
    sys.path.insert(0, str(up / "lot"))


def stats(ours, theirs, mask=None):
    a, b = ours.double().flatten(), theirs.double().flatten()
    if mask is not None:
        a, b = a[mask.flatten()], b[mask.flatten()]
    d = a - b
    rel = float(d.norm() / b.norm().clamp(min=1e-300))
    cos = float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-300))
    return {"rel": rel, "cos": cos, "max_abs": float(d.abs().max()), "n": int(a.numel())}


# ---- fixtures -------------------------------------------------------------------------------------------------

def fixtures(a):
    from mixed_res import TwoTrackPlan, region_layout, region_mask
    from h3_musubi import _layout, audio_frames, sigma_schedule
    from musubi_tuner.minimax_h3.packing import build_position_grid

    cases = []
    for name, text_len, latent_t, lat_h, lat_w, seed in [("clip_1344x768_22f", 37, 7, 24, 42, 1), ("still_768x448", 11, 1, 14, 24, 2),
                                                         ("clip_768x512_39f", 23, 12, 16, 24, 3)]:
        g = torch.Generator().manual_seed(seed)
        th, tw = lat_h // 2, lat_w // 2
        # a field with a blob: smooth scores the way a head's are
        yy, xx = torch.meshgrid(torch.arange(th).float(), torch.arange(tw).float(), indexing="ij")
        blob = torch.exp(-((yy - th * 0.4) ** 2 + (xx - tw * 0.55) ** 2) / (2 * (0.2 * th) ** 2))
        scores = (4 * blob[None].repeat(latent_t, 1, 1) + 0.7 * torch.randn(latent_t, th, tw, generator=g)).float()
        mask = region_mask(scores, alpha=0.5)
        plan = TwoTrackPlan(mask)
        layout = _layout(text_len, latent_t, th * 2, tw * 2)
        hr_layout = _layout(text_len, latent_t, 4 * th, 4 * tw)
        lr_count = plan.lr_count
        positions = build_position_grid(layout)[0]
        hr_positions = build_position_grid(hr_layout)[0][-4 * lr_count:]
        packed = torch.cat([positions, hr_positions[plan.hr_index]])
        full = mask.repeat_interleave(2, 1).repeat_interleave(2, 2)
        lot = region_layout(full, background=4, ring=1)
        cases.append({
            "name": name, "text_len": text_len, "latent_t": latent_t, "lat_h": lat_h, "lat_w": lat_w, "audio_t": audio_frames(latent_t),
            "scores": scores.flatten().tolist(), "mask": [int(v) for v in mask.flatten().tolist()],
            "hr_index": plan.hr_index.tolist(), "cell_order": plan.cell_order.tolist(), "grid_source": plan.grid_source("cpu").tolist(),
            "positions": packed.flatten().tolist(),
            "lot": [[r.t, r.u, r.v, r.eh, r.ew] for r in lot.rects],
        })
    sig = sigma_schedule(20)
    tail = torch.linspace(float(sig[16]), 0.0, 7, dtype=torch.float64)
    out = {"cases": cases, "sigmas20": sig.tolist(), "tail_16_6": tail.tolist()}
    Path(a.out).write_text(json.dumps(out))
    print(f"fixtures: {len(cases)} cases -> {a.out}")


# ---- the int8 checkpoint as float32 -----------------------------------------------------------------------------

def hadamard(g=256):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    m = torch.ones(1, 1)
    while m.shape[0] < g:
        m = torch.kron(m, h4)
    return m / (g ** 0.5)


class Int8File:
    """The ComfyUI int8 ConvRot checkpoint: W = (q * scale) H per 256 group (reference/regen2k/dit_ref.py)"""

    def __init__(self, path):
        self.f = safe_open(path, "pt")

    def get(self, name):
        return self.f.get_tensor(name)

    def linear(self, name):
        w = self.f.get_tensor(name + ".weight")
        if w.dtype != torch.int8:
            return w.float()
        q = w.float() * self.f.get_tensor(name + ".weight_scale").float()
        n, k = q.shape
        return (q.view(n, k // 256, 256) @ H).view(n, k)


def block_state(ck, i):
    p = f"blocks.{i}"
    sd = {f"{l}.weight": ck.linear(f"{p}.{l}") for l in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")}
    for n in ("norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight", "adaln_proj.linear.weight", "adaln_proj.linear.bias"):
        sd[n] = ck.get(f"{p}.{n}").float()
    return sd


def load_model(path, blocks=None):
    """musubi's MiniMaxH3Model in float32, built empty; the small parts loaded, each block loaded before it runs and
    emptied after (one block in memory at a time). `blocks`: run only the first N (a partial check)."""
    from musubi_tuner.minimax_h3.checkpoint import resolve_safetensors_files
    from musubi_tuner.minimax_h3.model import MiniMaxH3Model, classify_h3_transformer

    files = resolve_safetensors_files(path)
    config = classify_h3_transformer(files, allow_convrot_int8=True)
    with torch.device("meta"):
        dit = MiniMaxH3Model(config, dtype=torch.float32)
    ck = Int8File(str(path))

    def put(module, sd):
        module.to_empty(device="cpu")
        missing, unexpected = module.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert not missing, missing

    put(dit.video_patch_proj, {"weight": ck.get("video_patch_proj.weight").float(), "bias": ck.get("video_patch_proj.bias").float()})
    put(dit.audio_patch_proj, {"weight": ck.get("audio_patch_proj.weight").float(), "bias": ck.get("audio_patch_proj.bias").float()})
    dit.rope.to_empty(device="cpu")
    dit.rope.inv_freq.copy_(ck.get("rope.inv_freq").float())
    dit._buffers["adaln_t_table"] = ck.get("adaln_t_table").float()
    fl = dit.final_layer
    put(fl, {"norm.weight": ck.get("final_layer.norm.weight").float(), "adaln_proj.linear.weight": ck.get("final_layer.adaln_proj.linear.weight").float(),
             "adaln_proj.linear.bias": ck.get("final_layer.adaln_proj.linear.bias").float(),
             "video_out.weight": ck.get("final_layer.video_out.weight").float(), "video_out.bias": ck.get("final_layer.video_out.bias").float(),
             "audio_out.weight": ck.get("final_layer.audio_out.weight").float(), "audio_out.bias": ck.get("final_layer.audio_out.bias").float()})
    # the text arrives refined (the engine's dump): no projection, no refiner
    dit.condition_proj = torch.nn.Identity()
    dit.token_refiner = torch.nn.Identity()
    n = len(dit.blocks)
    limit = n if blocks is None else blocks
    t_load = [0.0]

    def pre(i):
        def hook(module, args):
            t0 = time.time()
            module.to_empty(device="cpu")
            missing, unexpected = module.load_state_dict(block_state(ck, i), strict=False)
            assert not missing and not unexpected, (missing, unexpected)
            t_load[0] += time.time() - t0
        return hook

    def post(module, args, out):
        module.to_empty(device="meta")
        return out

    for i, b in enumerate(dit.blocks):
        b.register_forward_pre_hook(pre(i))
        b.register_forward_hook(post)
    if limit < n:
        dit.blocks = torch.nn.ModuleList(list(dit.blocks)[:limit])
    return dit, ck, t_load


# ---- the checks ---------------------------------------------------------------------------------------------------

def check(a):
    global H
    H = hadamard()
    torch.set_num_threads(a.threads)
    from mixed_res import RegionHead, TwoTrackPlan, UpMaps, region_mask, two_track_attention
    import h3_musubi
    import two_track_h3
    from musubi_tuner.minimax_h3.packing import build_position_grid

    d = load_file(a.dump)
    with safe_open(a.dump, "pt") as f:
        meta = f.metadata()
    sigma = float(meta["sigma"])
    prefix = int(meta["prefix"])
    text_len = int(meta["text_len"])
    t, th, tw = d["mask"].shape
    rep = {"dump": a.dump, "sigma": sigma, "tokens": {"prefix": prefix, "half_res": t * th * tw, "full_res": int(d["hr_index"].numel())}}

    # 1. the region head on the same block-25 features, and the mask
    with safe_open(a.head, "pt") as f:
        hm = f.metadata()
    head = RegionHead(int(hm["hidden"]))
    head.load_state_dict(load_file(a.head))
    with torch.no_grad():
        scores = head(d["states"]).reshape(t, th, tw)
    ours_mask = d["mask"].bool()
    up_mask = region_mask(scores, alpha=0.5)
    up_mask_ours_scores = region_mask(d["scores"].float(), alpha=0.5)
    rep["region_head"] = {"scores": stats(d["scores"], scores), "mask_cells": int(ours_mask.numel()),
                          "mask_differ": int((up_mask != ours_mask).sum()), "mask_differ_same_scores": int((up_mask_ours_scores != ours_mask).sum()),
                          "share": float(ours_mask.float().mean()), "block": int(hm["block"])}
    print("region head:", json.dumps(rep["region_head"]), flush=True)

    # 2. the plan and the packed positions (integers exactly; positions as float32, the precision of the rotation)
    plan = TwoTrackPlan(ours_mask)
    layout = h3_musubi._layout(text_len, t, 2 * th, 2 * tw)
    hr_layout = h3_musubi._layout(text_len, t, 4 * th, 4 * tw)
    positions = build_position_grid(layout)[0]
    hr_positions = build_position_grid(hr_layout)[0][-4 * plan.lr_count:]
    packed = torch.cat([positions, hr_positions[plan.hr_index]])
    rep["layout"] = {"hr_index_equal": bool(torch.equal(plan.hr_index, d["hr_index"].long())),
                     "prefix_equal": layout.target_video_segment.start == prefix,
                     "positions_rows": int(packed.shape[0]), "positions_equal_f32": bool(torch.equal(packed.float(), d["positions"].float())),
                     "positions_max_abs": float((packed - d["positions"].double()).abs().max())}
    print("layout:", json.dumps(rep["layout"]), flush=True)

    # 3. the K/V map on the same keys (block 0, the half-res keys outside M, normalized, before rotation)
    with safe_open(a.maps, "pt") as f:
        mm = f.metadata()
    maps = UpMaps(int(mm["layers"]), 128, [(1, 2, 2)], rank=int(mm["rank"]))
    maps.load_state_dict(load_file(a.maps))
    with torch.no_grad():
        keys = d["knorm0"].unsqueeze(0)
        exp32 = maps.expand(0, (1, 2, 2), "k", keys)[0].reshape(d["expanded_k0"].shape)
        exp16 = maps.expand(0, (1, 2, 2), "k", keys.bfloat16())[0].float().reshape(d["expanded_k0"].shape)
    rep["kv_map"] = {"ours_host_vs_f32": stats(d["expanded_k0"], exp32), "upstream_bf16_vs_f32": stats(exp16, exp32),
                     "delta_share": float((exp32 - keys[0].unsqueeze(1)).norm() / exp32.norm())}
    print("kv map:", json.dumps(rep["kv_map"]), flush=True)

    if not a.int8:
        Path(a.report).write_text(json.dumps(rep, indent=1))
        print(f"no --int8: the attention and the step skipped; report -> {a.report}")
        return
    dit, ck, t_load = load_model(a.int8, a.blocks)

    # 4. block 0's dual-resolution attention on the engine's own qkv (raw, as its qkv linear wrote it)
    qkv = d["qkv0"]
    rows = qkv.shape[0]
    attn = dit.blocks[0].attn
    attn.to_empty(device="cpu")
    attn.q_norm.weight.data.copy_(ck.get("blocks.0.attn.q_norm.weight").float())
    attn.k_norm.weight.data.copy_(ck.get("blocks.0.attn.k_norm.weight").float())

    class Fixed(torch.nn.Module):
        def forward(self, x):
            return qkv.unsqueeze(0)

    keep_qkv, keep_out = attn.qkv_proj, attn.out_proj
    attn.qkv_proj, attn.out_proj = Fixed(), torch.nn.Identity()
    with torch.no_grad():
        rot = dit._rotation_table(packed.unsqueeze(0), torch.float32)
        hr_rot = dit._rotation_table(hr_positions.unsqueeze(0), torch.float32)
        att = two_track_attention(attn, torch.zeros(1, rows, 1), rot, layer=0, plan=plan, maps=maps, video_start=prefix, hr_rotation=hr_rot)[0]
    attn.qkv_proj, attn.out_proj = keep_qkv, keep_out
    split = prefix + plan.lr_count
    ours_att = d["att0"][:rows]
    rep["attention_block0"] = {"all_rows": stats(ours_att, att), "half_res_rows": stats(ours_att[:split], att[:split]),
                               "full_res_rows": stats(ours_att[split:], att[split:])}
    print("attention:", json.dumps(rep["attention_block0"]), flush=True)
    dit.blocks[0].to_empty(device="meta")

    # 5. the whole two-track call: upstream's two_track_forward on the dump's inputs (the sound as the engine fed it)
    if not a.no_step:
        audio = d["audio"].unsqueeze(0)
        h3_musubi._silence = lambda *args, **kw: audio
        lr, hr = d["lr"].unsqueeze(0), d["hr"].unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            v_lr, v_hr = two_track_h3.two_track_forward(dit, lr, hr, plan, 1.0 - sigma, d["text"].unsqueeze(0), maps=maps)
        secs = time.time() - t0
        inside = plan.mask.repeat_interleave(4, 1).repeat_interleave(4, 2)[None, None].expand_as(v_hr)
        # the engine's velocity is upstream's negated (as in every H3 path here)
        rep["two_track_step"] = {"blocks": len(dit.blocks), "seconds": secs, "load_seconds": t_load[0],
                                 "v_half_res": stats(-d["v_lr"].unsqueeze(0), v_lr),
                                 "v_full_res_region": stats(-d["v_hr"].unsqueeze(0), v_hr, inside.bool()),
                                 "v_full_res_outside_zero": float(d["v_hr"].unsqueeze(0)[~inside.bool()].abs().max()),
                                 "per_frame_full_res_cos": [stats(-d["v_hr"][:, k], v_hr[0, :, k], inside[0, 0, k].expand_as(v_hr[0, :, k]).bool())["cos"]
                                                            for k in range(t) if plan.mask[k].any()]}
        print("step:", json.dumps(rep["two_track_step"]), flush=True)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(f"report -> {a.report}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("fixtures", "check"))
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", default="mpeg_fixture.json")
    ap.add_argument("--int8")
    ap.add_argument("--maps")
    ap.add_argument("--head")
    ap.add_argument("--dump")
    ap.add_argument("--report", default="tt_ref_report.json")
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--no-step", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    setup(a.upstream)
    {"fixtures": fixtures, "check": check}[a.phase](a)


if __name__ == "__main__":
    main()
