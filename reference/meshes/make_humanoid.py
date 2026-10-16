"""make_humanoid.py: the riggers' test mesh, built procedurally (no assets, no licence): a humanoid in T-pose made of
capsules (torso, neck, head, arms, hands, legs, feet), Y up, feet near y = 0, facing +Z, with a cylindrical UV map per
part (seams split in the OBJ: one position, several UVs). Pure Python (no numpy).

    python3 make_humanoid.py > humanoid.obj         (7.6K vertices)
    python3 make_humanoid.py lo > humanoid_lo.obj   (3.6K vertices: under AnimateAnyMesh's 4096, so its padding runs)
"""
import math
import sys


def frame(d):
    """Two unit vectors perpendicular to unit d (and to each other)"""
    a = (1.0, 0.0, 0.0) if abs(d[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = (d[1] * a[2] - d[2] * a[1], d[2] * a[0] - d[0] * a[2], d[0] * a[1] - d[1] * a[0])
    n = math.sqrt(sum(x * x for x in u))
    u = tuple(x / n for x in u)
    w = (d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0])
    return u, w


LO = len(sys.argv) > 1 and sys.argv[1] == "lo"


def capsule(a, b, r, squash=(1.0, 1.0), seg=None, cap=None, mid=None):
    """A capsule from a to b of radius r (squashed across by `squash` along its two side axes):
    (positions, uvs, quads of (position, uv) corners)"""
    seg, cap, mid = (seg or (16 if LO else 24)), (cap or (6 if LO else 8)), (mid or (4 if LO else 6))
    d = tuple(b[i] - a[i] for i in range(3))
    length = math.sqrt(sum(x * x for x in d))
    d = tuple(x / length for x in d)
    u, w = frame(d)
    rings = []  # (centre, offset along the axis, radius scale)
    rings.append((a, -r, 0.0))                       # the pole at a
    for k in range(1, cap):                           # the cap at a
        t = -math.pi / 2 + math.pi / 2 * k / cap
        rings.append((a, math.sin(t) * r, math.cos(t)))
    for k in range(mid + 1):                          # the cylinder
        c = tuple(a[i] + (b[i] - a[i]) * k / mid for i in range(3))
        rings.append((c, 0.0, 1.0))
    for k in range(1, cap):                           # the cap at b
        t = math.pi / 2 * k / cap
        rings.append((b, math.sin(t) * r, math.cos(t)))
    rings.append((b, r, 0.0))                         # the pole at b
    pos, uv, corner = [], [], {}
    nr = len(rings)
    for ri, (c, along, s) in enumerate(rings):
        pole_at = None
        for si in range(seg + 1):
            if s == 0.0 and pole_at is not None:
                pi = pole_at
            elif si == seg:
                pi = corner[(ri, 0)][0]               # the seam: the ring's first position, another UV
            else:
                ph = 2 * math.pi * si / seg
                cu, cw = math.cos(ph) * s * r * squash[0], math.sin(ph) * s * r * squash[1]
                pos.append(tuple(c[i] + d[i] * along + u[i] * cu + w[i] * cw for i in range(3)))
                pi = len(pos) - 1
                if s == 0.0:
                    pole_at = pi
            uv.append((si / seg, ri / (nr - 1)))
            corner[(ri, si)] = (pi, len(uv) - 1)
    quads = []
    for ri in range(nr - 1):
        for si in range(seg):
            quads.append((corner[(ri, si)], corner[(ri, si + 1)], corner[(ri + 1, si + 1)], corner[(ri + 1, si)]))
    return pos, uv, quads


def humanoid():
    parts = [
        capsule((0, 0.92, 0), (0, 1.40, 0), 0.17, (1.0, 0.62)),        # torso
        capsule((0, 1.42, 0), (0, 1.56, 0), 0.055),                     # neck
        capsule((0, 1.62, 0.01), (0, 1.74, 0.01), 0.11),                # head
    ]
    for s in (-1, 1):
        parts += [
            capsule((s * 0.17, 1.40, 0), (s * 0.47, 1.40, 0), 0.055),   # upper arm
            capsule((s * 0.47, 1.40, 0), (s * 0.74, 1.40, 0), 0.045),   # forearm
            capsule((s * 0.76, 1.40, 0), (s * 0.86, 1.40, 0), 0.045, (0.45, 1.0)),  # hand
            capsule((s * 0.10, 0.92, 0), (s * 0.10, 0.50, 0), 0.08),    # thigh
            capsule((s * 0.10, 0.50, 0), (s * 0.10, 0.09, 0), 0.06),    # shin
            capsule((s * 0.10, 0.05, -0.03), (s * 0.10, 0.05, 0.16), 0.045, (1.0, 0.8)),  # foot
        ]
    return parts


def main():
    out = sys.stdout
    out.write("# a procedural humanoid (reference/meshes/make_humanoid.py): Y up, T-pose, capsules with UVs\n")
    vbase, tbase = 1, 1
    faces = []
    for pos, uv, quads in humanoid():
        for p in pos:
            out.write("v %.6f %.6f %.6f\n" % p)
        for t in uv:
            out.write("vt %.6f %.6f\n" % t)
        for q in quads:
            uniq = []
            for c in q:   # a quad at a pole has two corners on one position: a triangle
                if all(c[0] != x[0] for x in uniq):
                    uniq.append(c)
            if len(uniq) >= 3:
                faces.append([(vbase + c[0], tbase + c[1]) for c in uniq])
        vbase += len(pos)
        tbase += len(uv)
    for f in faces:
        out.write("f " + " ".join("%d/%d" % c for c in f) + "\n")


if __name__ == "__main__":
    main()
