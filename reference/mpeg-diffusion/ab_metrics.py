"""mpegsing/ab_metrics.py: the singer A/B's measurements for one clip (CPU only).

  python -I ab_metrics.py CLIP.mp4 REPORT.json VOCALS.wav GUIDE.mp4 ANCHOR.png OUT.json [--carried 22] [--region-from MPEG_REPORT.json]

- lip sync: mouth opening per frame (MediaPipe face landmarker: inner-lip gap 13-14 over mouth width 78-308) against the
  vocal stem's loudness per frame (RMS in dB over each 1/24 s), Pearson r at lags -8..8 frames over the frames after
  the carried ones (lag > 0: the mouth moves after the sound); also the same against a time-reversed envelope (a null)
- face: Laplacian variance of luma inside the face box (landmarks' extent) and inside the mouth box; ArcFace (insightface
  buffalo_l w600k_r50, onnxruntime CPU) cosine of each new frame's face to the anchor picture's face and to the carried
  frames' (the guide's singer: their mean embedding)
- carried frames: PSNR of frames 0..carried-1 against the guide's last frames (cover-fitted as the engine reads them),
  the whole frame and the face box; the seam: mean |gradient| of luma on the region's edge against 3 px either side
  (the MPEG run's region, also measured on the normal clip at the same place as a control)
"""
import json
import sys

import cv2
import numpy as np
import soundfile as sf

# the weights, each in its own directory: mp/face_landmarker.task (MediaPipe) and arcface/w600k_r50.onnx (insightface buffalo_l)
W = "/var/home/maxfridbe/src/mpegsing/weights"


def frames_of(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def cover(img, w, h):
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (round(iw * s), round(ih * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=f"{W}/mp/face_landmarker.task"), running_mode=vision.RunningMode.IMAGE,
                                        num_faces=1, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
    lm = vision.FaceLandmarker.create_from_options(opts)

    def once(rgb, x0, y0, size):
        # a square crop at (x0, y0), enlarged to 512 px: the detector looks at small faces too
        h, w = rgb.shape[:2]
        x0, y0 = int(max(0, min(x0, w - size))), int(max(0, min(y0, h - size)))
        c = cv2.resize(rgb[y0:y0 + size, x0:x0 + size], (512, 512), interpolation=cv2.INTER_CUBIC)
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(c)))
        if not res.face_landmarks:
            return None
        return np.array([[x0 + p.x * size, y0 + p.y * size] for p in res.face_landmarks[0]], dtype=np.float64)

    last = [None]

    def run(rgb):
        h, w = rgb.shape[:2]
        p = None
        if last[0] is not None:
            # around the face of the frame before: a crop 3.5 face widths across
            c = last[0].mean(0)
            size = int(min(h, max(160, 3.5 * np.ptp(last[0][:, 0]))))
            p = once(rgb, c[0] - size / 2, c[1] - size / 2, size)
        if p is None:
            # no face to follow: every window at two sizes, the largest face found (a sign's letters can pass for a small one)
            found = []
            for size in (min(h, 384), h):
                for y0 in range(0, h - size + 1, max(1, size // 2)):
                    for x0 in range(0, w - size + 1, size // 2):
                        q = once(rgb, x0, y0, size)
                        if q is not None:
                            found.append(q)
            p = max(found, key=lambda q: np.ptp(q[:, 0]), default=None)
        last[0] = p
        return p

    def reset():
        last[0] = None
    run.reset = reset
    return run


ARC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def arcface():
    import onnxruntime as ort
    s = ort.InferenceSession(f"{W}/arcface/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name

    def emb(rgb, pts):
        # five points: eye centres, nose tip, mouth corners (MediaPipe indices)
        le = pts[[33, 133]].mean(0)
        re = pts[[362, 263]].mean(0)
        five = np.array([le, re, pts[1], pts[61], pts[291]], dtype=np.float32)
        m, _ = cv2.estimateAffinePartial2D(five, ARC, method=cv2.LMEDS)
        face = cv2.warpAffine(rgb, m, (112, 112), borderValue=0.0)
        x = (face.astype(np.float32) - 127.5) / 127.5
        e = s.run(None, {name: x.transpose(2, 0, 1)[None]})[0][0]
        return e / np.linalg.norm(e)
    return emb


def luma(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def lapvar(y, box):
    x0, y0, x1, y1 = box
    c = y[max(0, y0):y1, max(0, x0):x1]
    if c.size < 64:
        return float("nan")
    return float(cv2.Laplacian(c.astype(np.float64), cv2.CV_64F).var())


def box_of(pts, idx=None, pad=0.15):
    p = pts if idx is None else pts[idx]
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return int(x0 - dx), int(y0 - dy), int(x1 + dx), int(y1 + dy)


MOUTH = [61, 291, 0, 17, 13, 14, 78, 308, 81, 311, 178, 402]


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(m, 1e-10)))


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 8 or a.std() == 0 or b.std() == 0:
        return float("nan"), int(len(a))
    return float(np.corrcoef(a, b)[0, 1]), int(len(a))


def main():
    a = sys.argv[1:]
    clip, report, vocals, guide, anchor, out = a[:6]
    carried = int(a[a.index("--carried") + 1]) if "--carried" in a else 22
    region_from = a[a.index("--region-from") + 1] if "--region-from" in a else report
    fr = frames_of(clip)
    n = len(fr)
    h, w = fr[0].shape[:2]
    rep = json.load(open(report))
    # the vocal stem's loudness per video frame (its own sample rate; the clip's frame k covers [k, k+1) / 24 s)
    v, sr = sf.read(vocals, always_2d=True)
    v = v.mean(1)
    hop = sr / 24.0
    env = []
    for k in range(n):
        s = v[int(k * hop):int((k + 1) * hop)]
        env.append(10 * np.log10(np.mean(s ** 2) + 1e-10) if len(s) else np.nan)
    env = np.array(env)
    lm = landmarker()
    emb = arcface()
    pts = [lm(f) for f in fr]
    anc = cover(cv2.cvtColor(cv2.imread(anchor), cv2.COLOR_BGR2RGB), w, h)
    lm.reset()
    pa = lm(anc)
    e_anchor = emb(anc, pa) if pa is not None else None
    # the identity reference: the carried frames' faces (the guide's singer), their mean embedding
    embs = [emb(f, p) if p is not None else None for f, p in zip(fr, pts)]
    ref = [e for e in embs[:carried] if e is not None]
    e0 = (lambda m: m / np.linalg.norm(m))(np.mean(ref, 0)) if ref else None
    mar, sharp_face, sharp_mouth, id_anchor, id_first = [], [], [], [], []
    for f, p in zip(fr, pts):
        if p is None:
            mar.append(np.nan); sharp_face.append(np.nan); sharp_mouth.append(np.nan); id_anchor.append(np.nan); id_first.append(np.nan)
            continue
        width = np.linalg.norm(p[78] - p[308])
        mar.append(np.linalg.norm(p[13] - p[14]) / max(width, 1e-6))
        y = luma(f.astype(np.float64))
        sharp_face.append(lapvar(y, box_of(p, pad=0.05)))
        sharp_mouth.append(lapvar(y, box_of(p, MOUTH, pad=0.4)))
        e = embs[len(mar) - 1]
        id_anchor.append(float(e @ e_anchor) if e_anchor is not None else np.nan)
        id_first.append(float(e @ e0) if e0 is not None else np.nan)
    mar = np.array(mar)
    new = slice(carried, n)
    lags = {}
    for lag in range(-8, 9):
        # mouth at frame k against the sound at k - lag
        m = mar[new]
        idx = np.arange(carried, n) - lag
        ok = (idx >= 0) & (idx < n)
        lags[lag] = pearson(m[ok], env[idx[ok]])[0]
    # the best lag within +-4 frames (+-167 ms: a mouth leads or trails its sound by less)
    best = max((l for l in lags if abs(l) <= 4 and np.isfinite(lags[l])), key=lambda l: lags[l], default=0)
    null = pearson(mar[new], env[new][::-1])[0]
    # carried frames against the guide's last frames (cover-fitted, as the engine reads a clip guide)
    g = frames_of(guide)
    g = [cover(x, w, h) for x in g[-carried:]] if len(g) >= carried else []
    cps, cps_face = [], []
    for i in range(min(carried, len(g))):
        cps.append(psnr(fr[i], g[i]))
        p = pts[i]
        if p is not None:
            x0, y0, x1, y1 = box_of(p, pad=0.05)
            cps_face.append(psnr(fr[i][max(0, y0):y1, max(0, x0):x1], g[i][max(0, y0):y1, max(0, x0):x1]))
    # the seam: |gradient| on the region's edge (the MPEG run's mask, a half-res token = 64 x 64 px) over its neighbours
    seam = None
    rr = json.load(open(region_from)).get("mpeg") or {}
    mask = rr.get("region_mask")
    if mask:
        t = None
        th, tw = h // 64, w // 64
        t = len(mask) // (th * tw)
        mk = np.array(mask, dtype=np.uint8).reshape(t, th, tw)
        of = []
        for k in range(t):
            of += [k] * (1 if k % 5 == 0 else 4)
        of = (of + [t - 1] * n)[:n]
        ratios = {"carried": [], "new": []}
        for i in range(n):
            m = cv2.resize(mk[of[i]], (w, h), interpolation=cv2.INTER_NEAREST)
            if m.min() == m.max():
                continue
            edge = (cv2.dilate(m, np.ones((3, 3), np.uint8)) != cv2.erode(m, np.ones((3, 3), np.uint8)))
            near = (cv2.dilate(m, np.ones((9, 9), np.uint8)) != cv2.erode(m, np.ones((9, 9), np.uint8))) & ~edge
            y = luma(fr[i].astype(np.float64))
            gr = np.abs(cv2.Sobel(y, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(y, cv2.CV_64F, 0, 1))
            if edge.sum() and near.sum():
                ratios["carried" if i < carried else "new"].append(float(gr[edge].mean() / max(gr[near].mean(), 1e-6)))
        seam = {k: (float(np.mean(v)) if v else None) for k, v in ratios.items()}
    nanmean = lambda x: float(np.nanmean(x)) if np.isfinite(np.asarray(x, float)).any() else None
    res = {
        "clip": clip, "frames": n, "carried": carried, "faces_found": int(sum(p is not None for p in pts)),
        "lipsync": {"r_lag0": lags[0], "r_best": lags[best], "best_lag": best, "r_by_lag": lags, "r_null_reversed": null,
                    "mouth_open_mean": nanmean(mar[new]), "mouth_open_std": float(np.nanstd(mar[new]))},
        "face": {"sharp_face": nanmean(sharp_face[carried:]), "sharp_mouth": nanmean(sharp_mouth[carried:]),
                 "sharp_face_carried": nanmean(sharp_face[:carried]),
                 "id_anchor": nanmean(id_anchor[carried:]), "id_first": nanmean(id_first[carried:]), "id_anchor_min": float(np.nanmin(id_anchor[carried:])) if np.isfinite(id_anchor[carried:]).any() else None},
        "carried_frames": {"psnr": nanmean(cps), "psnr_face": nanmean(cps_face), "psnr_per_frame": cps, "psnr_frame11": cps[11] if len(cps) > 11 else None},
        "seam_edge_over_near": seam,
        "per_frame": {"mouth_open": [None if not np.isfinite(x) else float(x) for x in mar], "vocal_db": [float(x) for x in env],
                      "sharp_face": [None if not np.isfinite(x) else float(x) for x in sharp_face]},
        "times": {k: rep.get(k) for k in ["sample_seconds", "total_seconds", "text_encoder_seconds", "decode_seconds"]},
    }
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["frames", "faces_found", "lipsync", "face", "seam_edge_over_near", "times"]} | {"carried": {k: res["carried_frames"][k] for k in ["psnr", "psnr_face", "psnr_frame11"]}})[:2000])


if __name__ == "__main__":
    main()
