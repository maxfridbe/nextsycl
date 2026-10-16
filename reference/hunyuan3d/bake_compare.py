"""The engine's bake (bake_texture.npy, bake_seen.npy from `nextsycl model3d check DIR --stages bake`) against Tencent's
(texture.npy, trust.npy from ref_paint.py bake on the same atlas), both on the texels both saw - as stored, and with the
engine's texture resampled onto Tencent's texel grid (Tencent rasterizes the atlas with texel centres at
v (size - 1) + 1/2, the engine at v size + 1/2 - glTF's sampling - so the same texel index is up to a texel apart).

  python bake_compare.py DIR
"""
import sys

import numpy as np


def main():
    d = sys.argv[1]
    mine = np.load(f"{d}/bake_texture.npy")
    seen = np.load(f"{d}/bake_seen.npy") > 0.5
    ref = np.load(f"{d}/texture.npy")
    trust = np.load(f"{d}/trust.npy")[..., 0] > 0.5
    n = mine.shape[0]

    def report(name, a, m):
        both = m & trust
        x, y = a[both].ravel(), ref[both].ravel()
        rel = np.sqrt(((x - y) ** 2).sum() / (y ** 2).sum())
        cos = (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())
        print(f"{name:<24} rel rms {rel:.3e}  cosine {cos:.6f}  ({both.sum()} texels both saw)")

    report("as stored", mine, seen)
    # Tencent's texel (r, c) is the engine's (r n / (n - 1) - 1/2, c n / (n - 1) - 1/2): bilinear
    g = np.arange(n) * n / (n - 1) - 0.5
    i0 = np.clip(np.floor(g).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    f = np.clip(g - np.floor(g), 0, 1)
    rows = mine[i0] * (1 - f)[:, None, None] + mine[i1] * f[:, None, None]
    res = rows[:, i0] * (1 - f)[None, :, None] + rows[:, i1] * f[None, :, None]
    s = seen[i0][:, i0] & seen[i1][:, i1] & seen[i0][:, i1] & seen[i1][:, i0]
    report("on Tencent's grid", res, s)


if __name__ == "__main__":
    main()
