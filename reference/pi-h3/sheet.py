"""A run dump's frames (`images` [F, H, W, 3] in [0, 1]) side by side as one PNG, to look at beside the engine's.

    python -I sheet.py dump.safetensors out.png 0 10 21
"""
import sys

import numpy as np
from PIL import Image
from safetensors.numpy import load_file


def main():
    img = load_file(sys.argv[1])['images']
    frames = [img[int(i)] for i in sys.argv[3:]]
    Image.fromarray((np.concatenate(frames, 1).clip(0, 1) * 255 + 0.5).astype(np.uint8)).save(sys.argv[2])


if __name__ == '__main__':
    main()
