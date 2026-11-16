"""The text encoder's input for Open-H3-Regenerate-2K-IC, as Diffusers' MiniMaxH3Ref2VATextEncoderStep builds it - the
draft as <Video 1> read at 2 fps in frame pairs, each block labelled with its time, then the release's task prompt with
the caption - against the token ids and modality tags of the engine (`ids`, `tags` in a regen2k check dump). Only the
processor and the tokenizer: the token counts per vision block come from the video processor's grid.

    python -I presentation_ref.py --processor <release>/base_model/processor --code <release>/infer --draft draft.mp4 \
        --size 448x256 --frames 22 --caption "..." --dump check.safetensors
"""
import argparse
import sys

import numpy as np
import torch
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    for k in ('processor', 'code', 'draft', 'size', 'caption', 'dump'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--frames', type=int, default=124)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    from transformers import AutoProcessor
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3VideoReference
    from ref2va_ic import encode, media
    w, h = (int(x) for x in a.size.split('x'))
    media.FRAMES, media.MIN_REAL = a.frames, min(media.MIN_REAL, a.frames)
    pixels, real, _ = media.read_draft(a.draft, h, w)
    frames = (pixels * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous().numpy()
    proc = AutoProcessor.from_pretrained(a.processor)
    step = MiniMaxH3Ref2VATextEncoderStep()
    ref = MiniMaxH3VideoReference(frames=frames, fps=24.0)
    _, img, vid, stamps = step._gather_vision_features(proc, [ref], 24.0)
    prompt = encode.TASK + '\n\nScene description: ' + a.caption.strip()
    ids, tags = step._build_presentation(proc.tokenizer, prompt, [ref], img, vid, stamps)
    d = load_file(a.dump)
    ours_ids, ours_tags = d['ids'].long().tolist(), d['tags'].long().tolist()
    print(f'release: {len(ids)} tokens, {len(stamps[0])} vision blocks of {vid[0]} tokens, labels {stamps[0]}')
    print(f'engine : {len(ours_ids)} tokens')
    same = ids == ours_ids and tags == ours_tags
    print('token ids and tags identical' if same else 'DIFFERENT')
    # the vision placeholders: the release's are <|video_pad|>, the engine's <|image_pad|> (ComfyUI's); the text encoder
    # replaces both by the vision tower's rows, so the rest is what has to agree
    norm = lambda v: [151655 if x == 151656 else x for x in v]
    if not same and norm(ids) == norm(ours_ids) and tags == ours_tags:
        print(f'identical but for the vision placeholder token ({sum(x == 151656 for x in ids)} <|video_pad|> in the release, <|image_pad|> in the engine)')
    elif not same:
        n = next((i for i, (x, y) in enumerate(zip(ids, ours_ids)) if x != y), min(len(ids), len(ours_ids)))
        print('first difference at', n, ids[max(0, n - 3):n + 5], ours_ids[max(0, n - 3):n + 5])


if __name__ == '__main__':
    main()
