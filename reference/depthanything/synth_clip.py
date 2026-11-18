"""A synthetic source clip for video.control's checks (nothing filmed, nothing copyrighted): a sky and a ground plane
with a horizon, a disc that moves across and comes nearer (grows), a box that stays, stripes on the ground. Raw RGB24
frames on stdout:

    python -I synth_clip.py W H FRAMES | ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r 24 -i - -c:v libx264 -pix_fmt yuv420p clip.mp4
"""
import sys

import numpy as np


def frame(w, h, t, n):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    hor = h * 0.45
    sky = y < hor
    img[..., 0] = np.where(sky, 120 + 60 * y / hor, 90 + 70 * (y - hor) / (h - hor))
    img[..., 1] = np.where(sky, 170 + 40 * y / hor, 120 + 30 * (y - hor) / (h - hor))
    img[..., 2] = np.where(sky, 235 - 30 * y / hor, 70)
    ground = ~sky
    depth = np.maximum(y - hor, 1.0)
    stripes = ground & ((np.floor(500.0 / depth + t * 0.35) % 2) == 0)
    img[stripes] *= 0.8
    # a box on the ground, fixed
    bx0, bx1, by0, by1 = int(w * 0.62), int(w * 0.82), int(h * 0.50), int(h * 0.78)
    img[by0:by1, bx0:bx1] = [180, 70, 60]
    # a disc crossing left to right and growing as it comes nearer
    k = t / max(n - 1, 1)
    cx, cy, r = w * (0.15 + 0.5 * k), h * (0.55 + 0.1 * k), h * (0.07 + 0.12 * k)
    disc = (x - cx) ** 2 + (y - cy) ** 2 < r * r
    img[disc] = [240, 230, 60]
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    w, h, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    out = sys.stdout.buffer
    for t in range(n):
        out.write(frame(w, h, t, n).tobytes())
