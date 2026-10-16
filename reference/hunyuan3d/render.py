"""A quick look at meshes: each GLB rendered from four sides (front, right, back, top-front), shaded by its normals
(the base colour texture when it has one), on a CPU z-buffer, side by side in one PNG.

  python render.py OUT.png A.glb [B.glb ...] [--size 384]
"""
import argparse

import numpy as np
import trimesh
from PIL import Image


def rot(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rx @ ry


def render(mesh, r, size):
    v = mesh.vertices @ r.T
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    c, s = (lo + hi) / 2 @ r.T, (hi - lo).max() * 0.55
    p = (v - c) / s
    xy = np.stack([(p[:, 0] + 1) * size / 2, (1 - p[:, 1]) * size / 2], 1)
    z = p[:, 2]
    f = mesh.faces
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.clip(n @ np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8]), 0, 1)
    tex, uv = None, None
    vis = getattr(mesh, "visual", None)
    if vis is not None and getattr(vis, "uv", None) is not None and getattr(vis.material, "baseColorTexture", None) is not None:
        tex = np.asarray(vis.material.baseColorTexture.convert("RGB"), np.float32) / 255
        uv = np.asarray(vis.uv)
    img = np.full((size, size, 3), 0.15, np.float32)
    zb = np.full((size, size), -np.inf, np.float32)
    for t in range(len(f)):
        if n[t, 2] <= 0:
            continue
        a, b, cc = xy[f[t]]
        x0, x1 = int(max(min(a[0], b[0], cc[0]), 0)), int(min(max(a[0], b[0], cc[0]) + 1, size))
        y0, y1 = int(max(min(a[1], b[1], cc[1]), 0)), int(min(max(a[1], b[1], cc[1]) + 1, size))
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (b[0] - a[0]) * (cc[1] - a[1]) - (cc[0] - a[0]) * (b[1] - a[1])
        if abs(d) < 1e-12:
            continue
        w1 = ((gx - a[0]) * (cc[1] - a[1]) - (cc[0] - a[0]) * (gy - a[1])) / d
        w2 = ((b[0] - a[0]) * (gy - a[1]) - (gx - a[0]) * (b[1] - a[1])) / d
        w0 = 1 - w1 - w2
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        zz = w0 * z[f[t, 0]] + w1 * z[f[t, 1]] + w2 * z[f[t, 2]]
        sub = zb[y0:y1, x0:x1]
        m &= zz > sub
        if not m.any():
            continue
        sub[m] = zz[m]
        shade = 0.25 + 0.75 * light[t]
        if tex is not None:
            u = w0 * uv[f[t, 0], 0] + w1 * uv[f[t, 1], 0] + w2 * uv[f[t, 2], 0]
            vv = w0 * uv[f[t, 0], 1] + w1 * uv[f[t, 1], 1] + w2 * uv[f[t, 2], 1]
            th, tw = tex.shape[:2]
            col = tex[np.clip(((1 - vv) * th).astype(int), 0, th - 1), np.clip((u * tw).astype(int), 0, tw - 1)]
            img[y0:y1, x0:x1][m] = col[m] * (0.5 + 0.5 * shade)
        else:
            img[y0:y1, x0:x1][m] = np.array([0.8, 0.78, 0.72]) * shade
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--size", type=int, default=384)
    a = ap.parse_args()
    rows = []
    for path in a.meshes:
        m = trimesh.load(path, force="mesh")
        views = [rot(0, 0), rot(np.pi / 2, 0), rot(np.pi, 0), rot(0.5, 0.6)]
        rows.append(np.concatenate([render(m, r, a.size) for r in views], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(a.out)


if __name__ == "__main__":
    main()
