#!/usr/bin/env python3
"""Reference test: a picture to its prompt and back (image.describe -> image.generate), through the front door's API.

Five pictures made here, none downloaded: three drawn by this script (flat shapes on white, a barn under a sun, a
sign that reads "OPEN" - pure Python, no packages) and two rendered by the box's own image model from fixed prompts
and seeds (--render; kept in --out and reused). Each is described for the target (image.describe), the prompt drawn
again by the target's image model at the suggested size, and the copy compared with the original:

- **colour**: the intersection of their colour histograms (4 x 4 x 4 bins; 1 = the same colours in the same amounts);
- **layout**: the correlation of their 16 x 16 grey thumbnails (1 = the same light and dark in the same places);
- **judge** (optional, --judge HOST:PORT: a chat server that reads pictures, `/api/chat`): 0-10, "how closely picture 2
  reproduces picture 1".

The same scores for each original against another picture's copy are the baseline: what "unrelated" scores here.
Loose by design - a prompt cannot fix a composition pixel by pixel - so the checks are only that every prompt is
written, quotes the lettering of the sign, and that each copy scores above the baseline's mean on colour or layout.

    tests/api/test-describe.py --target z-image-turbo --render z-image-turbo
    tests/api/test-describe.py --describe-cli "dist/nextsycl image describe" --model qwen3.8-27b-q4-describe ...
    tests/api/test-describe.py --speed 512,1024,2048          # pp / tg of the describe for one picture at each size

--describe-cli runs `nextsycl image describe ... --json` (the same code as image.describe, against a chat server)
instead of the front door's method - for a build that is not the box's yet. Needs Python 3.9+ (no packages). Writes
into --out: the pictures, their copies, report.json. Exits 1 when a check fails.
"""

import argparse
import base64
import json
import math
import shlex
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

# ------------------------------------------------------------------ PNG, by hand (8-bit RGB / RGBA, not interlaced)


def write_png(path, w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb[y * w * 3:(y + 1) * w * 3]) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def read_png(data):
    """(width, height, RGB bytes) of an 8-bit RGB or RGBA PNG"""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    i, idat, w = 8, b"", 0
    while i < len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        t, d = data[i + 4:i + 8], data[i + 8:i + 8 + n]
        if t == b"IHDR":
            w, h, depth, ctype, _, _, inter = struct.unpack(">IIBBBBB", d)
            assert depth == 8 and ctype in (2, 6) and inter == 0, f"PNG depth {depth} type {ctype} interlace {inter}"
            bpp = 3 if ctype == 2 else 4
        elif t == b"IDAT":
            idat += d
        i += 12 + n
    raw = zlib.decompress(idat)
    stride = w * bpp
    out = bytearray(h * stride)
    prev = bytearray(stride)
    for y in range(h):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if f == 1:
            for x in range(bpp, stride):
                line[x] = (line[x] + line[x - bpp]) & 255
        elif f == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 255
        elif f == 3:
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + ((a + prev[x]) >> 1)) & 255
        elif f == 4:
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                b, c = prev[x], prev[x - bpp] if x >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        out[y * stride:(y + 1) * stride] = line
        prev = line
    if bpp == 4:
        out = bytearray(b for k in range(0, len(out), 4) for b in out[k:k + 3])
    return w, h, bytes(out)


# ------------------------------------------------------------------ the pictures drawn here


def canvas(w, h, colour):
    return bytearray(bytes(colour) * (w * h))


def put(img, w, x, y, c):
    k = (y * w + x) * 3
    img[k:k + 3] = bytes(c)


def barn(w=1024, h=768):
    """a red barn with a dark roof on green grass under a blue sky, a yellow sun at the top right"""
    img = bytearray()
    for y in range(h):
        if y < h * 0.62:
            t = y / (h * 0.62)
            row = bytes((int(70 + 90 * t), int(130 + 80 * t), int(220 + 30 * t)))
        else:
            row = bytes((60, 150, 60))
        img += row * w
    cx, cy, r = int(w * 0.8), int(h * 0.18), int(h * 0.09)
    for y in range(cy - r, cy + r):
        for x in range(cx - r, cx + r):
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                put(img, w, x, y, (250, 210, 40))
    x0, x1, y0, y1 = int(w * 0.3), int(w * 0.62), int(h * 0.42), int(h * 0.72)
    for y in range(y0, y1):
        for x in range(x0, x1):
            put(img, w, x, y, (180, 30, 30))
    for y in range(int(h * 0.25), y0):
        half = (y - h * 0.25) / (y0 - h * 0.25) * (x1 - x0) / 2 * 1.1
        for x in range(int((x0 + x1) / 2 - half), int((x0 + x1) / 2 + half)):
            put(img, w, x, y, (60, 40, 40))
    for y in range(int(h * 0.56), y1):
        for x in range(int(w * 0.43), int(w * 0.49)):
            put(img, w, x, y, (245, 245, 235))
    return w, h, img


def shapes(w=1024, h=1024):
    """a red circle at the left, a blue square at the right and a green triangle below them, on white"""
    img = canvas(w, h, (255, 255, 255))
    cx, cy, r = int(w * 0.28), int(h * 0.32), int(w * 0.17)
    for y in range(cy - r, cy + r):
        for x in range(cx - r, cx + r):
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                put(img, w, x, y, (220, 30, 40))
    for y in range(int(h * 0.16), int(h * 0.48)):
        for x in range(int(w * 0.56), int(w * 0.88)):
            put(img, w, x, y, (30, 70, 210))
    top, base = int(h * 0.58), int(h * 0.92)
    for y in range(top, base):
        half = (y - top) / (base - top) * w * 0.2
        for x in range(int(w / 2 - half), int(w / 2 + half)):
            put(img, w, x, y, (40, 170, 70))
    return w, h, img


FONT = {  # 5 x 7
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
}


def sign(w=1280, h=720, word="OPEN"):
    """yellow block letters reading OPEN on a dark navy board"""
    img = canvas(w, h, (20, 30, 70))
    s = 22
    total = len(word) * 6 * s - s
    x0, y0 = (w - total) // 2, (h - 7 * s) // 2
    for i, ch in enumerate(word):
        for r, bits in enumerate(FONT[ch]):
            for c, bit in enumerate(bits):
                if bit == "1":
                    for y in range(y0 + r * s, y0 + (r + 1) * s):
                        for x in range(x0 + (i * 6 + c) * s, x0 + (i * 6 + c + 1) * s):
                            put(img, w, x, y, (250, 200, 30))
    return w, h, img


# what the box's own model renders as the other two (fixed prompts and seeds: written by hand for this test)
RENDERS = {
    "teapot": ("A close-up product photograph of a glossy turquoise ceramic teapot on a light oak table, soft daylight from a "
               "window on the left, a white linen napkin folded beside it, a plain cream wall behind, shallow depth of field.", 1234),
    "lighthouse": ("A flat vector illustration of a white and red striped lighthouse on a dark grey rock at dusk, an orange and "
                   "purple gradient sky, three white seagulls, calm navy sea with simple wave lines, no text.", 4321),
}

# ------------------------------------------------------------------ the comparisons


def thumb(w, h, rgb, n=16):
    """n x n mean colours (box filter)"""
    out = []
    for j in range(n):
        y0, y1 = j * h // n, (j + 1) * h // n
        for i in range(n):
            x0, x1 = i * w // n, (i + 1) * w // n
            r = g = b = 0
            cnt = 0
            for y in range(y0, y1, max(1, (y1 - y0) // 8)):
                for x in range(x0, x1, max(1, (x1 - x0) // 8)):
                    k = (y * w + x) * 3
                    r, g, b, cnt = r + rgb[k], g + rgb[k + 1], b + rgb[k + 2], cnt + 1
            out.append((r / cnt, g / cnt, b / cnt))
    return out


def histogram(w, h, rgb, step=4):
    hist = [0] * 64
    n = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            k = (y * w + x) * 3
            hist[(rgb[k] >> 6) * 16 + (rgb[k + 1] >> 6) * 4 + (rgb[k + 2] >> 6)] += 1
            n += 1
    return [v / n for v in hist]


def scores(a, b):
    """colour (histogram intersection) and layout (16 x 16 grey correlation)"""
    colour = sum(min(x, y) for x, y in zip(a["hist"], b["hist"]))
    ga = [0.299 * r + 0.587 * g + 0.114 * bl for r, g, bl in a["thumb"]]
    gb = [0.299 * r + 0.587 * g + 0.114 * bl for r, g, bl in b["thumb"]]
    ma, mb = sum(ga) / len(ga), sum(gb) / len(gb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ga, gb))
    va, vb = math.sqrt(sum((x - ma) ** 2 for x in ga)), math.sqrt(sum((y - mb) ** 2 for y in gb))
    layout = cov / (va * vb) if va > 0 and vb > 0 else 0.0
    return round(colour, 3), round(layout, 3)


def tint(a, b):
    """1 - the mean distance of the 16 x 16 thumbnails' colours (cell by cell, over the largest RGB distance): unlike the
    histogram's coarse bins, a navy a little darker still scores close"""
    d = sum(math.dist(x, y) for x, y in zip(a["thumb"], b["thumb"])) / len(a["thumb"])
    return round(1 - d / math.dist((0, 0, 0), (255, 255, 255)), 3)


def features(png):
    w, h, rgb = read_png(png)
    return {"hist": histogram(w, h, rgb), "thumb": thumb(w, h, rgb)}


# ------------------------------------------------------------------ the API


class Door:
    def __init__(self, url, token):
        self.url, self.token = url.rstrip("/"), token

    def rpc(self, method, body, timeout=3600):
        q = urllib.request.Request(f"{self.url}/rpc/{method}", json.dumps(body).encode(),
                                   {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        v = json.load(urllib.request.urlopen(q, timeout=timeout))
        if not v.get("ok"):
            raise RuntimeError(f"{method}: {v.get('error', {}).get('code')}: {v.get('error', {}).get('message')}")
        return v["result"]


def token_of(args):
    if args.token:
        return args.token
    try:
        return json.load(open(args.token_file))["token"]
    except OSError:
        sys.exit(f"no token: --token, or a readable {args.token_file}")


def describe(args, door, png_path, target, **extra):
    if args.describe_cli:
        cmd = shlex.split(args.describe_cli) + [str(png_path)] + shlex.split(args.describe_args or "") + ["--json", "--target", target]
        if args.model:
            cmd += ["--model", args.model]
        for k, v in extra.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"describe: {r.stderr.strip()[-400:]}")
        return json.loads(r.stdout)
    body = {"image": base64.b64encode(Path(png_path).read_bytes()).decode(), "target": target, "label": f"describe-test-{png_path.stem}-{int(time.time())}"}
    if args.model:
        body["model"] = args.model
    body.update({{"max_pixels": "max_pixels"}.get(k, k): v for k, v in extra.items()})
    return door.rpc("image.describe", body)


GEN_URL = None


def generate(door, model, prompt, size, seed):
    if GEN_URL:
        # an image server of its own (`nextsycl image serve`): OpenAI's images route
        q = urllib.request.Request(f"{GEN_URL}/v1/images/generations", json.dumps({"model": model, "prompt": prompt, "size": size, "seed": seed,
                                   "response_format": "b64_json"}).encode(), {"Content-Type": "application/json"})
        return base64.b64decode(json.load(urllib.request.urlopen(q, timeout=1800))["data"][0]["b64_json"])
    r = door.rpc("image.generate", {"model": model, "prompt": prompt, "size": size, "seed": seed, "response_format": "b64_json",
                                    "label": f"describe-test-gen-{seed}-{int(time.time())}"})
    return base64.b64decode(r["data"][0]["b64_json"])


def judge(server, a, b):
    """0-10 from a chat model that reads pictures (its /api/chat), or None"""
    body = {"stream": False, "max_tokens": 8, "temperature": 0, "think_budget": 0,
            "messages": [{"role": "user", "content": "Picture 1 is an original; picture 2 was drawn from a text description of it. "
                          "Score from 0 to 10 how closely picture 2 reproduces picture 1 in subject, layout, colours and style "
                          "(10: the same picture, 0: unrelated). Answer with the number only.",
                          "images": [base64.b64encode(a).decode(), base64.b64encode(b).decode()]}]}
    q = urllib.request.Request(f"http://{server}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        t = json.load(urllib.request.urlopen(q, timeout=600)).get("content", "")
        return float("".join(c for c in t if c.isdigit() or c == ".")[:4])
    except Exception as e:  # noqa: BLE001 - a judge that fails is reported, not fatal
        print(f"  judge: {e}", file=sys.stderr)
        return None


def resized(src_png, size, out):
    """the barn drawn again at size x size (the speed test's pictures)"""
    w, h, rgb = barn(size, size)
    write_png(out, w, h, rgb)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--token")
    ap.add_argument("--token-file", default="/var/lib/nextsycl/serve.json")
    ap.add_argument("--target", default="z-image-turbo", help="the prompt's target: a family or an image model's id")
    ap.add_argument("--draw", default="z-image-turbo", help="the image model that draws the prompts again")
    ap.add_argument("--render", help="render the two model pictures with this image model (else: the ones in --out)")
    ap.add_argument("--model", help="the describer (default: the door's choice)")
    ap.add_argument("--describe-cli", help="run this command (+ PICTURE --json ...) instead of the door's image.describe")
    ap.add_argument("--describe-args", help="more options for --describe-cli, after the picture (--server HOST:PORT ...)")
    ap.add_argument("--gen-url", help="http://HOST:PORT of an image server (nextsycl image serve) to draw with instead of the door")
    ap.add_argument("--judge", help="HOST:PORT of a chat server that reads pictures: a 0-10 score each")
    ap.add_argument("--speed", help="sizes (512,1024,2048): pp / tg of a describe at each, twice each; nothing drawn")
    ap.add_argument("--stage", choices=["all", "render", "describe", "draw", "score"], default="all",
                    help="one part only (a box where the describer and the image model take turns on one card): render the two "
                         "model pictures; describe (prompts.json); draw the copies from prompts.json; score (the copies there, the judge)")
    ap.add_argument("--only", help="the pictures to run (barn,shapes,sign,teapot,lighthouse)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="describe-test")
    args = ap.parse_args()
    global GEN_URL
    GEN_URL = args.gen_url
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # the door: for what it does here (describing without --describe-cli, drawing without --gen-url)
    draws = not args.speed and args.stage in ("all", "render", "draw")
    door = Door(args.url, token_of(args)) if (not args.describe_cli and args.stage != "render") or (draws and not args.gen_url) else None

    if args.speed:
        rows = []
        for n in [int(x) for x in args.speed.split(",")]:
            p = resized(None, n, out / f"speed-{n}.png")
            for k in range(2):
                r = describe(args, door, p, args.target, max_pixels=max(65536, min(4194304, n * n)))
                t = r["timings"]
                rows.append({"size": n, "run": k + 1, "seen": r["seen"], "prompt_tokens": r["tokens"]["prompt"], "reused": r["tokens"]["reused"],
                             "generated": r["tokens"]["generated"], "pp_tok_s": t["pp_tok_s"], "tg_tok_s": t["tg_tok_s"],
                             "prompt_s": t["prompt_seconds"], "generate_s": t["generate_seconds"], "seconds": t["seconds"]})
                print(json.dumps(rows[-1]))
        (out / "speed.json").write_text(json.dumps(rows, indent=1))
        return 0

    pics = {}
    for name, fn in (("barn", barn), ("shapes", shapes), ("sign", sign)):
        p = out / f"{name}.png"
        if not p.exists():
            write_png(p, *fn())
        pics[name] = p
    for name, (prompt, seed) in RENDERS.items():
        p = out / f"{name}.png"
        if args.render and not p.exists():
            p.write_bytes(generate(door, args.render, prompt, "1024x1024", seed))
        if p.exists():
            pics[name] = p
    if args.only:
        pics = {k: v for k, v in pics.items() if k in args.only.split(",")}

    if args.stage == "render":
        print("rendered:", ", ".join(sorted(pics)))
        return 0
    # the stages, for a box where one card holds one model at a time: describe (prompts.json), draw (the copies), score
    saved = out / "prompts.json"
    described = json.loads(saved.read_text()) if args.stage in ("draw", "score") and saved.exists() else {}
    report, failed = [], []
    for name, p in pics.items():
        t0 = time.time()
        r = described.get(name) or describe(args, door, p, args.target)
        described[name] = r
        prompt = r["prompt"]
        print(f"== {name}: {r['words']} words, pp {r['timings']['pp_tok_s']} tok/s, tg {r['timings']['tg_tok_s']} tok/s, {r['timings']['seconds']} s\n{prompt}")
        if args.stage == "describe":
            continue
        cp = out / f"{name}.copy.png"
        if args.stage != "score" or not cp.exists():
            cp.write_bytes(generate(door, args.draw, prompt, r["suggested_size"]["size"], args.seed))
        copy = cp.read_bytes()
        report.append({"picture": name, "prompt": prompt, "negative_prompt": r.get("negative_prompt"), "target": r["target"], "model": r["model"],
                       "size": r["suggested_size"]["size"], "timings": r["timings"], "tokens": r["tokens"], "seconds": round(time.time() - t0, 1),
                       "judge": judge(args.judge, p.read_bytes(), copy) if args.judge and args.stage != "draw" else None})
        if not prompt.strip():
            failed.append(f"{name}: no prompt")
        if name == "sign" and "OPEN" not in prompt:
            failed.append("sign: the prompt does not quote OPEN")
    if args.stage == "describe":
        saved.write_text(json.dumps(described, indent=1))
        return 1 if failed else 0
    feats = {row["picture"]: (features(pics[row["picture"]].read_bytes()), features((out / f"{row['picture']}.copy.png").read_bytes())) for row in report}
    names = [row["picture"] for row in report]
    for row in report:
        a, c = feats[row["picture"]]
        row["colour"], row["layout"] = scores(a, c)
        others = [scores(a, feats[o][1]) for o in names if o != row["picture"]]
        row["baseline_colour"] = round(sum(x[0] for x in others) / max(1, len(others)), 3)
        row["baseline_layout"] = round(sum(x[1] for x in others) / max(1, len(others)), 3)
        row["tint"] = tint(a, c)
        row["baseline_tint"] = round(sum(tint(a, feats[o][1]) for o in names if o != row["picture"]) / max(1, len(others)), 3)
        if others and row["colour"] <= row["baseline_colour"] and row["layout"] <= row["baseline_layout"]:
            failed.append(f"{row['picture']}: its copy scores no better than other pictures' copies")
        print(f"{row['picture']:10s} colour {row['colour']:.3f} (baseline {row['baseline_colour']:.3f})  tint {row['tint']:.3f} (baseline {row['baseline_tint']:.3f})  "
              f"layout {row['layout']:+.3f} (baseline {row['baseline_layout']:+.3f})  judge {row['judge']}")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    for f in failed:
        print("FAIL", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
