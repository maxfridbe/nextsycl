"""Frames of a draft and its 2K regeneration side by side, and how sharp each is: frames N of both (the draft enlarged
to the 2K size with bicubic), a crop of each at full 2K resolution, and the variance of the Laplacian of the luma (a
sharpness measure: higher is more fine detail) of the draft enlarged and of the regeneration.

    python -I look.py --draft draft.mp4 --regen regen.mp4 --frames 10,60,115 --crop X,Y,W,H --out DIR
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def frame(path, n, w, h):
    """frame n of a clip at w x h (bicubic): a .png beside it named <clip>.<n>.png when there (a host whose ffmpeg has
    no H.264 decoder: extract them where one is), else ffmpeg"""
    png = Path(f'{path}.{n}.png')
    if png.exists():
        return np.asarray(Image.open(png).convert('RGB').resize((w, h), Image.BICUBIC))
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'select=eq(n\\,{n}),scale={w}:{h}:flags=bicubic', '-frames:v', '1', '-f', 'rawvideo',
                          '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def sharpness(img):
    y = img.astype(np.float64) @ [0.299, 0.587, 0.114]
    lap = -4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]
    return float(lap.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draft', required=True)
    ap.add_argument('--regen', required=True)
    ap.add_argument('--frames', default='10,60,115')
    ap.add_argument('--size', default='2560x1440')
    ap.add_argument('--crop', default='1000,250,640,480')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split('x'))
    cx, cy, cw, ch = (int(x) for x in a.crop.split(','))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in (int(x) for x in a.frames.split(',')):
        d, r = frame(a.draft, n, w, h), frame(a.regen, n, w, h)
        print(f'frame {n}: sharpness draft (enlarged) {sharpness(d):.1f}, regen {sharpness(r):.1f}; '
              f'crop {sharpness(d[cy:cy + ch, cx:cx + cw]):.1f} vs {sharpness(r[cy:cy + ch, cx:cx + cw]):.1f}; mean abs diff {np.abs(d.astype(int) - r).mean():.1f}')
        Image.fromarray(np.concatenate([d, r], 1)).resize((w, h // 2)).save(out / f'pair_{n}.png')
        Image.fromarray(np.concatenate([d[cy:cy + ch, cx:cx + cw], r[cy:cy + ch, cx:cx + cw]], 1)).save(out / f'crop_{n}.png')


if __name__ == '__main__':
    main()
