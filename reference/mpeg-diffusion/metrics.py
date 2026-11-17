"""Two clips of the same prompt, seed and steps compared: MPEG-Diffusion's two-track clip against H3 at full resolution
(docs/ports/mpeg-diffusion.md). The region is the two-track run's own (its job report's mpeg.region_mask, the half-res
token grid: one token = 64 x 64 pixels, each latent frame over the pixel frames it decodes to).

    python3 -I metrics.py --full full.mp4 --tt tt.mp4 --report tt.json [--lpips] [--png DIR --at 10,40,70]

Per clip: PSNR and SSIM (luma, Gaussian 11 x 11, sigma 1.5) over the whole frame and inside the region; LPIPS (AlexNet,
the lpips package, CPU) whole frame and inside the region (its spatial map averaged there), every 4th frame; flicker:
the mean absolute frame-to-frame change of luma (0-255) of each clip, whole and inside the region; sharpness: the
variance of the luma Laplacian inside and outside the region; the region's share of the frame. --png writes frame
pairs (full | two-track, the region outlined in the latter) to look at.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def read(path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path],
                           capture_output=True, check=True)
    s = json.loads(probe.stdout)["streams"][0]
    w, h = s["width"], s["height"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def luma(f):
    f = f.astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def gauss_filter(x, sigma=1.5, radius=5):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    x = np.apply_along_axis(lambda r: np.convolve(np.pad(r, radius, mode="reflect"), k, mode="valid"), 1, x)
    return np.apply_along_axis(lambda c: np.convolve(np.pad(c, radius, mode="reflect"), k, mode="valid"), 0, x)


def ssim_map(a, b):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = gauss_filter(a), gauss_filter(b)
    va, vb = gauss_filter(a * a) - ma * ma, gauss_filter(b * b) - mb * mb
    cov = gauss_filter(a * b) - ma * mb
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


def psnr(a, b, m=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = d[m].mean() if m is not None else d.mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


def laplacian_var(y, m):
    lap = -4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]
    mm = m[1:-1, 1:-1]
    return float(lap[mm].var()) if mm.any() else float("nan")


def frame_of(t, frames):
    v = [k for k in range(t) for _ in range(1 if k % 5 == 0 else 4)]
    return (v + [v[-1]] * frames)[:frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--tt", required=True)
    ap.add_argument("--report", required=True, help="the two-track job's JSON (its mpeg.region_mask)")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--png")
    ap.add_argument("--at", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    full, tt = read(a.full), read(a.tt)
    n = min(len(full), len(tt))
    full, tt = full[:n], tt[:n]
    rep = json.loads(Path(a.report).read_text())
    m = rep.get("mpeg", rep)
    h, w = full.shape[1:3]
    th, tw = h // 64, w // 64
    mask = np.array(m["region_mask"], dtype=bool)
    t = mask.size // (th * tw)
    mask = mask.reshape(t, th, tw)
    fo = frame_of(t, n)
    region = np.stack([np.kron(mask[fo[f]], np.ones((64, 64), dtype=bool)).astype(bool) for f in range(n)])
    out = {"frames": n, "size": f"{w}x{h}", "region_share": float(mask.mean())}
    yf, yt = luma(full), luma(tt)
    out["psnr_frame"] = psnr(full, tt)
    out["psnr_region"] = psnr(full, tt, np.repeat(region[..., None], 3, -1))
    ss_all, ss_reg = [], []
    for f in range(0, n, 2):
        s = ssim_map(yf[f], yt[f])
        ss_all.append(s.mean())
        if region[f].any():
            ss_reg.append(s[region[f]].mean())
    out["ssim_frame"] = float(np.mean(ss_all))
    out["ssim_region"] = float(np.mean(ss_reg)) if ss_reg else None
    df, dt = np.abs(np.diff(yf, axis=0)), np.abs(np.diff(yt, axis=0))
    rr = region[1:] & region[:-1]
    out["flicker_full"] = float(df.mean())
    out["flicker_tt"] = float(dt.mean())
    out["flicker_full_region"] = float(df[rr].mean()) if rr.any() else None
    out["flicker_tt_region"] = float(dt[rr].mean()) if rr.any() else None
    # the change of the change: how far the two-track clip's frame-to-frame change is from the full-res clip's
    out["flicker_tt_vs_full_mad"] = float(np.abs(dt - df).mean())
    sh = {"full_region": [], "tt_region": [], "full_outside": [], "tt_outside": []}
    for f in range(0, n, 4):
        sh["full_region"].append(laplacian_var(yf[f], region[f]))
        sh["tt_region"].append(laplacian_var(yt[f], region[f]))
        sh["full_outside"].append(laplacian_var(yf[f], ~region[f]))
        sh["tt_outside"].append(laplacian_var(yt[f], ~region[f]))
    out["sharpness"] = {k: float(np.nanmean(v)) for k, v in sh.items()}
    if a.lpips:
        import lpips
        import torch
        net = lpips.LPIPS(net="alex", spatial=True, verbose=False)
        la, lr_ = [], []
        with torch.no_grad():
            for f in range(0, n, 4):
                x = torch.from_numpy(full[f]).permute(2, 0, 1)[None].float() / 127.5 - 1
                y = torch.from_numpy(tt[f]).permute(2, 0, 1)[None].float() / 127.5 - 1
                d = net(x, y)[0, 0].numpy()
                la.append(d.mean())
                if region[f].any():
                    lr_.append(d[region[f]].mean())
        out["lpips_frame"] = float(np.mean(la))
        out["lpips_region"] = float(np.mean(lr_)) if lr_ else None
    if a.png:
        from PIL import Image, ImageDraw
        Path(a.png).mkdir(parents=True, exist_ok=True)
        for f in [int(x) for x in a.at.split(",") if x]:
            f = min(f, n - 1)
            right = Image.fromarray(tt[f])
            dr = ImageDraw.Draw(right)
            g = mask[fo[f]]
            for u in range(th):
                for v in range(tw):
                    if not g[u, v]:
                        continue
                    x, y = v * 64, u * 64
                    if u == 0 or not g[u - 1, v]:
                        dr.line([(x, y), (x + 64, y)], fill="white", width=2)
                    if u == th - 1 or not g[u + 1, v]:
                        dr.line([(x, y + 64), (x + 64, y + 64)], fill="white", width=2)
                    if v == 0 or not g[u, v - 1]:
                        dr.line([(x, y), (x, y + 64)], fill="white", width=2)
                    if v == tw - 1 or not g[u, v + 1]:
                        dr.line([(x + 64, y), (x + 64, y + 64)], fill="white", width=2)
            pair = Image.new("RGB", (2 * w, h))
            pair.paste(Image.fromarray(full[f]), (0, 0))
            pair.paste(right, (w, 0))
            pair.resize((w, h // 2)).save(Path(a.png) / f"pair_{f}.png")
            Image.fromarray(tt[f]).save(Path(a.png) / f"tt_{f}.png")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
