"""How close a clip's frames are to its reference pictures, by image embeddings: DINOv3 ViT-H+/16 (the CLS token,
appearance and identity) and SigLIP so400m (pooled; HunyuanVideo-1.5's vision tower) - the cosine of each frame with each
reference, averaged over the frames, for clips made from the same shot under different models.

    python -I refsim.py --dino DIR --siglip DIR --frames DIR --refs DIR shot1:ref_a.png,ref_b.png ...

Frames are PNGs `<clip>_NN.png` in --frames (cut from the clip beforehand, past its carried-over opening); one line a clip
and reference, and the mean over the clip's references.
"""
import argparse
import glob
import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel


def main():
    ap = argparse.ArgumentParser()
    for k in ('dino', 'siglip', 'frames', 'refs'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('shots', nargs='+', help='CLIP_PREFIX:ref1.png,ref2.png')
    a = ap.parse_args()
    torch.set_num_threads(8)
    enc = []
    dp = AutoImageProcessor.from_pretrained(a.dino)
    dm = AutoModel.from_pretrained(a.dino).eval()
    enc.append(('dino', dp, lambda x: dm(pixel_values=x).pooler_output))
    sp = AutoImageProcessor.from_pretrained(a.siglip)
    sm = SiglipVisionModel.from_pretrained(a.siglip).eval()
    enc.append(('siglip', sp, lambda x: sm(pixel_values=x).pooler_output))

    @torch.no_grad()
    def emb(paths):
        out = {}
        imgs = [Image.open(p).convert('RGB') for p in paths]
        for name, proc, f in enc:
            e = f(proc(images=imgs, return_tensors='pt')['pixel_values'])
            out[name] = torch.nn.functional.normalize(e.float(), dim=-1)
        return out

    print('| clip | reference | DINOv3 | SigLIP |')
    print('|---|---|---|---|')
    for s in a.shots:
        clip_prefix, refs = s.split(':')
        refs = refs.split(',')
        r = emb([os.path.join(a.refs, x) for x in refs])
        for clip in sorted({os.path.basename(p).rsplit('_', 1)[0] for p in glob.glob(os.path.join(a.frames, clip_prefix + '*_??.png'))}):
            fr = sorted(glob.glob(os.path.join(a.frames, clip + '_??.png')))
            f = emb(fr)
            cos = {n: (f[n] @ r[n].T).mean(0) for n in f}            # per reference, over the frames
            for j, ref in enumerate(refs):
                print(f'| {clip} | {ref} | {float(cos["dino"][j]):.3f} | {float(cos["siglip"][j]):.3f} |')
            print(f'| {clip} | **mean** | **{float(cos["dino"].mean()):.3f}** | **{float(cos["siglip"].mean()):.3f}** |', flush=True)


if __name__ == '__main__':
    main()
