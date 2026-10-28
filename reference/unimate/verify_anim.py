"""Reads a GLB with a skin and a skeletal animation (what `nextsycl model3d rig RIGGED.glb --model unimate` writes) and
checks the animation does what a viewer would show: it poses the mesh by linear blend skinning in numpy for each frame
and reports how the body moves (bounding box, root travel, how far each joint turns, whether anything explodes).

    python -I verify_anim.py anim.glb [--dump out.npy]    (--dump: the skinned positions [frames, vertices, 3])
"""
import json
import struct
import sys

import numpy as np


def load(path):
    b = open(path, "rb").read()
    assert b[:4] == b"glTF"
    jl, jt = struct.unpack("<II", b[12:20])
    doc = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl, bt = struct.unpack("<II", b[off:off + 8])
    return doc, b[off + 8:off + 8 + bl]


def accessor(doc, bin_, i):
    a = doc["accessors"][i]
    v = doc["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
    dt = {5126: "<f4", 5123: "<u2", 5125: "<u4", 5121: "u1"}[a["componentType"]]
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(bin_, dtype=dt, count=a["count"] * n, offset=start).reshape(a["count"], n)


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    doc, bin_ = load(sys.argv[1])
    skin = doc["skins"][0]
    joints = skin["joints"]
    nodes = doc["nodes"]
    prim = doc["meshes"][0]["primitives"][0]
    pos = accessor(doc, bin_, prim["attributes"]["POSITION"]).astype(np.float64)
    jnt = accessor(doc, bin_, prim["attributes"]["JOINTS_0"]).astype(int)
    wgt = accessor(doc, bin_, prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    ibm = accessor(doc, bin_, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)   # column-major -> row-major
    anim = next(a for a in doc["animations"] if a["name"] == "skeleton")
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    rot, trans, times = {}, {}, None
    for ch in anim["channels"]:
        s = anim["samplers"][ch["sampler"]]
        times = accessor(doc, bin_, s["input"])[:, 0]
        out = accessor(doc, bin_, s["output"])
        (rot if ch["target"]["path"] == "rotation" else trans)[ch["target"]["node"]] = out
    frames = len(times)
    print(f"{len(joints)} joints, {frames} frames over {times[-1]:.2f} s, {len(pos)} vertices; rotation tracks {len(rot)}, translation tracks {len(trans)}")
    norms = np.concatenate([np.linalg.norm(r, axis=1) for r in rot.values()])
    print(f"quaternion lengths {norms.min():.6f} .. {norms.max():.6f}")

    def world(t):
        cache = {}

        def w(i):
            if i in cache:
                return cache[i]
            n = nodes[i]
            m = np.eye(4)
            m[:3, 3] = trans[i][t] if i in trans else n.get("translation", [0, 0, 0])
            m[:3, :3] = quat_mat(rot[i][t] if i in rot else n.get("rotation", [0, 0, 0, 1]))
            cache[i] = (w(parent[i]) @ m) if i in parent else m
            return cache[i]
        return [w(j) for j in joints]

    posed = np.zeros((frames, len(pos), 3))
    for t in range(frames):
        mats = [m @ ibm[k] for k, m in enumerate(world(t))]
        mats = np.stack(mats)
        h = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
        acc = np.zeros((len(pos), 3))
        for k in range(4):
            acc += wgt[:, k:k + 1] * np.einsum("vij,vj->vi", mats[jnt[:, k]], h)[:, :3]
        posed[t] = acc
    lo, hi = posed.min(axis=(0, 1)), posed.max(axis=(0, 1))
    rest = pos.max(axis=0) - pos.min(axis=0)
    print(f"rest box {np.round(rest, 3)}; posed box over the clip {np.round(hi - lo, 3)}")
    c0 = posed[0].mean(axis=0)
    cs = posed.mean(axis=1)
    print(f"centroid path: start {np.round(c0, 3)}, end {np.round(cs[-1], 3)}, travel {np.round(cs[-1] - c0, 3)}")
    motion = np.abs(np.diff(posed, axis=0)).mean()
    print(f"mean vertex step per frame {motion:.5f}; frame 0 vs rest {np.abs(posed[0] - pos).mean():.5f}")
    print(f"posed vertices finite: {np.isfinite(posed).all()}; largest |coordinate| {np.abs(posed).max():.2f}")
    ang = {}
    for j, r in rot.items():
        a = 2 * np.arccos(np.clip(np.abs(r[:, 3]), 0, 1))
        ang[j] = a
    top = sorted(ang.items(), key=lambda kv: -kv[1].max())[:6]
    print("joints that turn most (peak degrees):", ", ".join(f"{nodes[j].get('name')}={np.degrees(a.max()):.0f}" for j, a in top))
    if "--dump" in sys.argv:
        np.save(sys.argv[sys.argv.index("--dump") + 1], posed.astype(np.float32))


main()
