"""inspect_glb.py: what a rigged or animated GLB holds, in numbers - the riggers' and animators' outputs checked
without a viewer. Pure Python + numpy.

    python inspect_glb.py OUT.glb

Prints the mesh's size; for a skin: the joints (name, parent, head) and how many vertices each one leads; for a morph
animation: each frame's largest and mean displacement, and where on the body (by height band) the motion is.
"""
import json
import struct
import sys

import numpy as np


def read(path):
    b = open(path, "rb").read()
    assert b[:4] == b"glTF"
    jl = struct.unpack("<I", b[12:16])[0]
    doc = json.loads(b[20:20 + jl])
    bin_at = 20 + jl + 8
    blob = b[bin_at:]

    def acc(i):
        a = doc["accessors"][i]
        v = doc["bufferViews"][a["bufferView"]]
        dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a["componentType"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(blob, dtype=dt, count=a["count"] * n, offset=off).reshape(a["count"], n)

    return doc, acc


def main():
    doc, acc = read(sys.argv[1])
    prim = doc["meshes"][0]["primitives"][0]
    pos = acc(prim["attributes"]["POSITION"])
    print("vertices %d, triangles %d" % (len(pos), len(acc(prim["indices"])) // 3))
    lo, hi = pos.min(0), pos.max(0)
    print("bounds", lo.round(3), hi.round(3))
    if "JOINTS_0" in prim["attributes"]:
        j = acc(prim["attributes"]["JOINTS_0"])
        w = acc(prim["attributes"]["WEIGHTS_0"])
        skin = doc["skins"][0]
        names = [doc["nodes"][n]["name"] for n in skin["joints"]]
        lead = np.bincount(j[np.arange(len(j)), w.argmax(1)], minlength=len(names))
        print("joints %d" % len(names))
        parent = {}
        for i, n in enumerate(doc["nodes"]):
            for c in n.get("children", []):
                parent[c] = i
        ibm = acc(skin["inverseBindMatrices"]).reshape(-1, 4, 4)
        for k, n in enumerate(skin["joints"]):
            head = -ibm[k][3, :3]
            p = parent.get(n)
            print("  %-12s parent %-12s head %s leads %d vertices" % (names[k], doc["nodes"][p]["name"] if p is not None and p in skin["joints"] else "-",
                                                                     head.round(3), lead[k]))
        print("weights: rows sum %.4f..%.4f, mean largest %.3f" % (w.sum(1).min(), w.sum(1).max(), w.max(1).mean()))
    if "targets" in prim:
        h = pos[:, 1]
        bands = np.linspace(h.min(), h.max(), 6)
        for f, t in enumerate(prim["targets"]):
            d = np.linalg.norm(acc(t["POSITION"]), axis=1)
            per = [d[(h >= bands[i]) & (h <= bands[i + 1])].mean() for i in range(5)]
            print("frame %2d: max %.4f mean %.4f; by height (low to high) %s" % (f, d.max(), d.mean(), " ".join("%.4f" % x for x in per)))


if __name__ == "__main__":
    main()
