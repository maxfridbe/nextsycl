"""The reference for nextsycl's Depth Anything V2 Small engine (image/depthanything), on the CPU in float32: what
transformers' DepthAnythingForDepthEstimation makes of a synthetic picture, every stage dumped for
`nextsycl image depth --check DIR`.

    python -I ref.py --model DIR_WITH_model.safetensors_and_config.json --out DIR [--pictures DIR]

Environment: pip install torch transformers numpy "pillow==12.2.0" safetensors (CPU torch is enough). Pillow is
pinned like rembg's reference: nextsycl's Picture resizes across first, as Pillow up to 12.2 does.

Without --pictures it draws three synthetic ones (no photographs, nothing copyrighted): 640x360 (landscape, the
position embeddings interpolated), 518x518 (the model's own grid), 300x500 (portrait). Per picture NAME (DIR/NAME/):
  picture.png    the picture (RGB)
  input.npy      [h, w, 3] float32: PIL bicubic to the size the rule below gives, ImageNet mean/std
  emb.npy        [1 + N, 384] after the embeddings (cls, then N = (h/14)(w/14) patches)
  tap0..3.npy    [1 + N, 384] the layers 3, 6, 9, 12 outputs through the backbone's final layer norm
  fused3.npy     [h/14*2^3.., ..., 64] the last fusion layer's output (channels last)
  pred.npy       [h, w] the network's output (the head's ReLU), h and w the input's
  depth.npy      [H, W] float32: pred resized to the picture's own size, bilinear, align_corners False
  depth8.png     depth.npy scaled to its own [min, max] x 255 (truncated)
The input size rule is DPTImageProcessor's: size 518 x 518, keep_aspect_ratio, ensure_multiple_of 14 - the scale
nearest to 1 on either side is applied to both, each side rounded to a multiple of 14.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def synth(w, h, seed):
    """A deterministic picture with structure at several scales: a sky-to-ground gradient, a ground plane, a disc, a
    box, a sinusoidal band and fine stripes."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 90 + 120 * y / h
    img[..., 1] = 140 + 60 * np.sin(x / w * 3.0)
    img[..., 2] = 220 - 150 * y / h
    ground = y > h * 0.62
    img[ground] = np.stack([110 + 40 * np.sin(x[ground] / 17.0), 90 + 30 * y[ground] / h, 70 + 0 * x[ground]], -1)
    disc = (x - w * 0.3) ** 2 + (y - h * 0.55) ** 2 < (min(w, h) * 0.18) ** 2
    img[disc] = [235, 60, 50]
    img[int(h * 0.45):int(h * 0.8), int(w * 0.6):int(w * 0.85)] = [40, 90, 200]
    band = np.abs(y - (h * 0.2 + 12 * np.sin(x / 23.0))) < 6
    img[band] = [250, 250, 120]
    stripes = (y > h * 0.88) & ((x // 5) % 2 == 0)
    img[stripes] *= 0.55
    img += rng.normal(0, 3.0, img.shape).astype(np.float32)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def input_size(h, w, target=518, multiple=14):
    sh, sw = target / h, target / w
    if abs(1 - sw) < abs(1 - sh):
        sh = sw
    else:
        sw = sh
    r = lambda v: round(v / multiple) * multiple
    return r(sh * h), r(sw * w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pictures")
    a = ap.parse_args()
    from transformers import DepthAnythingForDepthEstimation

    model = DepthAnythingForDepthEstimation.from_pretrained(a.model).eval().float()
    if a.pictures:
        names = sorted(os.listdir(a.pictures))
        pics = [(os.path.splitext(n)[0], Image.open(os.path.join(a.pictures, n)).convert("RGB")) for n in names]
    else:
        pics = [("land640x360", synth(640, 360, 1)), ("square518", synth(518, 518, 2)), ("portrait300x500", synth(300, 500, 3))]
    bb = model.backbone
    for name, im in pics:
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        im.save(os.path.join(d, "picture.png"))
        W, H = im.size
        h, w = input_size(H, W)
        x = np.asarray(im.resize((w, h), Image.BICUBIC), np.float32) / 255.0
        x = ((x - MEAN) / STD).astype(np.float32)
        np.save(os.path.join(d, "input.npy"), x)
        pv = torch.from_numpy(x).permute(2, 0, 1)[None]
        with torch.no_grad():
            emb = bb.embeddings(pv)
            np.save(os.path.join(d, "emb.npy"), emb[0].numpy())
            out = bb.forward_with_filtered_kwargs(pv, output_hidden_states=True) if hasattr(bb, "forward_with_filtered_kwargs") else bb(pv)
            # the taps through the backbone's layer norm, tokens kept (reshape_hidden_states false)
            for i, f in enumerate(out.feature_maps):
                np.save(os.path.join(d, f"tap{i}.npy"), f[0].numpy())
            ph, pw = h // 14, w // 14
            hs = model.neck(list(out.feature_maps), ph, pw)
            np.save(os.path.join(d, "fused3.npy"), hs[-1][0].permute(1, 2, 0).numpy())
            pred = model.head(hs, ph, pw)[0]
        np.save(os.path.join(d, "pred.npy"), pred.numpy())
        dep = F.interpolate(pred[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
        np.save(os.path.join(d, "depth.npy"), dep)
        mi, ma = dep.min(), dep.max()
        Image.fromarray(((dep - mi) / (ma - mi) * 255).astype(np.uint8)).save(os.path.join(d, "depth8.png"))
        print(name, (W, H), "->", (w, h), "pred", float(pred.min()), float(pred.max()))


if __name__ == "__main__":
    main()
