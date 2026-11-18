"""mpegsing/exact.py: the conditioning checks from the runs' dumps (ex.sh). python -I exact.py OUTDIR"""
import json
import sys

import numpy as np
from safetensors.numpy import load_file
from safetensors import safe_open

d = sys.argv[1]


def load(p):
    with safe_open(p, "np") as f:
        meta = f.metadata()
    return load_file(p), meta


def segs(meta, key="segments"):
    return [(s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for s in meta[key].split(",")]


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return "identical" if (a == b).all() else f"max |diff| {diff.max():.3g}, {int((diff > 0).sum())} of {a.size} differ"


out = {}
# E1: region empty. MPEG's half-res denoiser (its cond dump) against H3 at 672x384 (its cond dump)
m, mm = load(f"{d}/e1_mpeg.cond.safetensors")
h, hm = load(f"{d}/e1_half.cond.safetensors")
out["e1_layout_segments"] = "identical" if mm["segments"] == hm["segments"] else f"{mm['segments']} vs {hm['segments']}"
for k in sorted(h):
    out[f"e1 {k}"] = same(m[k], h[k])
lm = load_file(f"{d}/e1_mpeg.latents.safetensors")
lh = load_file(f"{d}/e1_half.latents.safetensors")
out["e1 half-res video latent (whole run) vs H3 672x384"] = same(lm["mpeg.half_res_video"], lh["samples.video"])
out["e1 audio latent (whole run) vs H3 672x384"] = same(lm["samples.audio"], lh["samples.audio"])
# E2: region everything. The switch's full-res keyframe rows against H3 at 1344x768's keyframe rows
s, sm = load(f"{d}/e2_mpeg.cond.switch.safetensors")
f, fm = load(f"{d}/e2_full.cond.safetensors")
out["e2 region share"] = float(s["mask"].mean())
out["e2 switch's full-res layout vs H3 1344x768's"] = "identical" if sm["hr_segments"] == fm["segments"] else f"{sm['hr_segments']} vs {fm['segments']}"
fs = segs(fm)
conds = [x for x in fs if x[0] in ("Cond", "CondAudio")]
vid = [(i, x) for i, x in enumerate(conds) if x[0] == "Cond"]
rows = s["cond_hr.rows"].astype(np.int64)
want_rows = np.concatenate([np.arange(x[1], x[2]) for _, x in vid])
out["e2 full-res keyframe rows: which"] = same(rows, want_rows)
want = np.concatenate([f[f"seg{i}.rows"] for i, _ in vid])
out["e2 full-res keyframe rows: patches (augmented)"] = same(s["cond_hr.patches"], want)
want = np.concatenate([f[f"seg{i}.positions"] for i, _ in vid])
out["e2 full-res keyframe rows: positions"] = same(s["cond_hr.positions"], want)
out["e2 expanded keyframe rows (should be none)"] = int(s["cond_out.rows"].size)
# the sound: MPEG's half-res audio keyframe rows against H3 1344x768's (latents are size-free; positions are not quite)
ma, mam = load(f"{d}/e2_mpeg.cond.safetensors")
ms = segs(mam)
mconds = [x for x in ms if x[0] in ("Cond", "CondAudio")]
for i, x in enumerate(conds):
    if x[0] == "CondAudio":
        j = [k for k, y in enumerate(mconds) if y[0] == "CondAudio"][0]
        out["sound keyframe rows (augmented) MPEG vs H3 1344x768"] = same(ma[f"seg{j}.rows"], f[f"seg{i}.rows"])
        out["sound keyframe timesteps"] = same(ma[f"seg{j}.t"], f[f"seg{i}.t"])
        p, q = ma[f"seg{j}.positions"], f[f"seg{i}.positions"]
        out["sound keyframe positions (time, height)"] = same(p[:, :2], q[:, :2])
        out["sound keyframe positions (width)"] = same(p[:, 2], q[:, 2]) + f"; values half-res {sorted(set(p[:, 2].tolist()))} full-res {sorted(set(q[:, 2].tolist()))}"
for i, x in vid:
    out[f"keyframe {i} timestep (H3 full res)"] = sorted(set(f[f"seg{i}.t"].tolist()))
print(json.dumps(out, indent=1))
