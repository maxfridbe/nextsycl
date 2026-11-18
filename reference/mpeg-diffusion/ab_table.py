"""The singer A/B table from ab_metrics.py's m_ab_*.json (markdown), run in the directory that holds them:

    python3 -I ab_table.py

The lip-sync null: the same mouth trace against the other chunks' vocal envelopes (best r within +-4 frames, averaged) -
what a mouth that does not follow this song scores.
"""
import json

import numpy as np

f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"


def best_r(mouth, env, carried, lags=range(-4, 5)):
    mouth, env = np.asarray(mouth, float), np.asarray(env, float)
    n = min(len(mouth), len(env))
    out = []
    for lag in lags:
        k = np.arange(carried, n)
        j = k - lag
        ok = (j >= 0) & (j < n)
        k, j = k[ok], j[ok]
        good = np.isfinite(mouth[k]) & np.isfinite(env[j])
        a, b = mouth[k[good]], env[j[good]]
        out.append(np.corrcoef(a, b)[0, 1] if len(a) > 8 and a.std() > 0 else np.nan)
    return np.nanmax(out)


ms = {}
for c in "ABCD":
    for s in (20, 10):
        for v in ("int8", "mpeg"):
            try:
                ms[(c, s, v)] = json.load(open(f"m_ab_{c}_s{s}_{v}.json"))
            except OSError:
                pass
env = {c: d["per_frame"]["vocal_db"] for (c, _, _), d in ms.items()}
print("| Chunk | Steps | Engine | Lip r lag 0 | Lip r best +-4 (lag) | Null: other songs' r | Face sharp | Mouth sharp | ID vs carried | ID vs anchor | "
      "Carried PSNR whole / face | Seam edge/near carried, new | Sampling + rest = total s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (c, s, v), d in ms.items():
    l, fc, cf, t = d["lipsync"], d["face"], d["carried_frames"], d["times"]
    se = d["seam_edge_over_near"] or {}
    m = [np.nan if x is None else x for x in d["per_frame"]["mouth_open"]]
    null = np.nanmean([best_r(m, env[o], d["carried"]) for o in env if o != c])
    tot, sam = t["total_seconds"], t["sample_seconds"]
    print(f"| {c} | {s} | {v.upper()} | {f(l['r_lag0'])} | {f(l['r_best'])} ({l['best_lag']}) | {null:.2f} | {f(fc['sharp_face'], 0)} | {f(fc['sharp_mouth'], 0)} | "
          f"{f(fc['id_first'])} | {f(fc['id_anchor'])} | {f(cf['psnr'], 1)} / {f(cf['psnr_face'], 1)} | {f(se.get('carried'))}, {f(se.get('new'))} | "
          f"{sam:.0f} + {tot - sam:.0f} = {tot:.0f} |")
