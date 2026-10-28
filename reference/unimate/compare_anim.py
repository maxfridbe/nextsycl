"""How far apart two animations of the same rig are (what dropping the joint names does to the motion): reads the
"skeleton" animation of two GLBs and reports, over all frames and joints, the mean and RMS rotation angle between
corresponding joints, each clip's own motion amplitude (RMS angle of the joints' rotations from their rest) and the
root travel.

    python -I compare_anim.py a.glb b.glb
"""
import json
import struct
import sys

import numpy as np


def load(path):
    b = open(path, "rb").read()
    jl, _ = struct.unpack("<II", b[12:20])
    doc = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl, _ = struct.unpack("<II", b[off:off + 8])
    return doc, b[off + 8:off + 8 + bl]


def accessor(doc, bin_, i):
    a = doc["accessors"][i]
    v = doc["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(bin_, dtype="<f4", count=a["count"] * n, offset=start).reshape(a["count"], n)


def tracks(path):
    doc, bin_ = load(path)
    anim = next(a for a in doc["animations"] if a["name"] == "skeleton")
    rot, root = {}, None
    for ch in anim["channels"]:
        out = accessor(doc, bin_, anim["samplers"][ch["sampler"]]["output"])
        name = doc["nodes"][ch["target"]["node"]].get("name")
        if ch["target"]["path"] == "rotation":
            rot[name] = out.astype(np.float64)
        else:
            root = out.astype(np.float64)
    return rot, root


def angle(a, b):
    d = np.abs(np.sum(a * b, axis=-1)).clip(0, 1)
    return 2 * np.arccos(d)


ra, ta = tracks(sys.argv[1])
rb, tb = tracks(sys.argv[2])
names = [n for n in ra if n in rb]
A = np.stack([ra[n] for n in names], axis=1)
B = np.stack([rb[n] for n in names], axis=1)
d = np.degrees(angle(A, B))
rest = np.array([0, 0, 0, 1.0])
amp = lambda X: np.degrees(np.sqrt(np.mean(angle(X, rest) ** 2)))
print(f"{len(names)} joints, {A.shape[0]} frames")
print(f"rotation between the two clips: mean {d.mean():.1f} deg, rms {np.sqrt((d ** 2).mean()):.1f} deg, worst joint-frame {d.max():.1f} deg")
print(f"motion amplitude (rms rotation from rest): a {amp(A):.1f} deg, b {amp(B):.1f} deg")
# the same measure against a rotation of the other clip by nothing: how big the difference is relative to the motion
print(f"difference / amplitude: {np.sqrt((d ** 2).mean()) / max(amp(A), 1e-9):.2f}")
if ta is not None and tb is not None:
    print(f"root travel: a {np.round(ta[-1] - ta[0], 2)}, b {np.round(tb[-1] - tb[0], 2)}; root path rms difference {np.sqrt(((ta - tb) ** 2).sum(axis=1).mean()):.2f}")
