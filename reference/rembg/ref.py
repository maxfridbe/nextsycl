"""The references for nextsycl's background removers (image/rembg), on the CPU in float32: for each picture in a folder,
what the model's own code makes of it, every stage dumped for `nextsycl image check`.

    python -I ref.py isnet --model DIR_WITH_isnet-general-use.onnx --pictures DIR --out DIR
    python -I ref.py birefnet --model BIREFNET_DIR --pictures DIR --out DIR

The environment: pip install rembg==2.0.67 onnxruntime opencv-python-headless "pillow<13" (birefnet: torch,
torchvision, transformers, timm, kornia, einops, opencv-python-headless, "pillow<13"). Pillow is pinned below 13:
13.0 runs a resize's vertical pass first when the height shrinks more than twice the width does (Resample.c's
horizontal_first), which moves rembg's Lanczos masks by a level or two on most photographs; 12.2 and before resize
across first, as nextsycl's Picture and nextcuda do (checked here with 12.2.0).

isnet: rembg 2.0.67's own session (isnet-general-use on onnxruntime's CPU provider) - nextcuda's reference, the same
calls. Dumps per picture NAME (DIR/NAME/): input.npy [1024, 1024, 3] float32 (rembg's normalize: Lanczos to 1024,
/ its max, - 0.5), pred.npy [1024, 1024] (the network's output_image, the sigmoid), mask.png (rembg's mask: scaled to
its range, 8 bits, Lanczos back), mask_pp.png (rembg's post_process on it), cutout.png (rembg.remove(...,
post_process_mask=True): the platform's call), picture.png (the picture as read, RGB).

birefnet: ZhengPeng7/BiRefNet's own code (transformers, trust_remote_code: BIREFNET_DIR holds birefnet.py,
BiRefNet_config.py, config.json and model.safetensors at the card's revision) as TRELLIS.2's rembg.BiRefNet runs it:
torchvision's Resize (bilinear) to 1024 x 1024, ImageNet normalization, the last prediction. Dumps per picture:
input.npy [1024, 1024, 3], logits.npy [1024, 1024], mask.png (the sigmoid as ToPILImage's 8 bits, PIL's bicubic
back), mask_pp.png (rembg's post_process on it), picture.png.
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def pictures(d):
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        yield os.path.splitext(os.path.basename(p))[0], Image.open(p)


def isnet(a):
    os.environ["U2NET_HOME"] = a.model
    from rembg import new_session, remove
    from rembg.bg import post_process
    s = new_session("isnet-general-use", providers=["CPUExecutionProvider"])
    for name, img in pictures(a.pictures):
        o = os.path.join(a.out, name)
        os.makedirs(o, exist_ok=True)
        img.convert("RGB").save(os.path.join(o, "picture.png"))
        inp = s.normalize(img, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (1024, 1024))
        x = list(inp.values())[0]
        np.save(os.path.join(o, "input.npy"), x[0].transpose(1, 2, 0).astype(np.float32))
        pred = s.inner_session.run(None, inp)[0][0, 0]
        np.save(os.path.join(o, "pred.npy"), pred.astype(np.float32))
        mask = s.predict(img)[0]
        mask.save(os.path.join(o, "mask.png"))
        Image.fromarray(post_process(np.array(mask))).save(os.path.join(o, "mask_pp.png"))
        remove(img, session=s, post_process_mask=True).save(os.path.join(o, "cutout.png"))
        print(name, img.size, flush=True)


def post_process(mask):
    """rembg 2.0.67's bg.post_process, line for line (the birefnet environment has torch, not rembg)"""
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    return np.where(mask < 127, 0, 255).astype(np.uint8)


def birefnet(a):
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    torch.set_num_threads(os.cpu_count() or 1)
    m = AutoModelForImageSegmentation.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32).eval().float()
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for name, img in pictures(a.pictures):
        o = os.path.join(a.out, name)
        os.makedirs(o, exist_ok=True)
        img = img.convert("RGB")
        img.save(os.path.join(o, "picture.png"))
        x = tf(img)[None]
        np.save(os.path.join(o, "input.npy"), x[0].permute(1, 2, 0).numpy().astype(np.float32))
        with torch.no_grad():
            logits = m(x)[-1]
        np.save(os.path.join(o, "logits.npy"), logits[0, 0].numpy().astype(np.float32))
        mask = transforms.ToPILImage()(logits.sigmoid()[0].squeeze()).resize(img.size)
        mask.save(os.path.join(o, "mask.png"))
        Image.fromarray(post_process(np.array(mask))).save(os.path.join(o, "mask_pp.png"))
        print(name, img.size, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["isnet", "birefnet"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--pictures", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    isnet(a) if a.which == "isnet" else birefnet(a)


if __name__ == "__main__":
    main()
