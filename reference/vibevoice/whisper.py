"""Transcribe WAV files with openai/whisper-small (transformers, CPU): the intelligibility check of nextsycl's speech
engines' clips.

    python whisper.py a.wav [b.wav ...] [--language english]
"""
import argparse

import numpy as np
from scipy.io import wavfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--language", default=None)
    a = ap.parse_args()
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
    for p in a.wavs:
        sr, w = wavfile.read(p)
        if w.dtype.kind == "i":
            w = w.astype(np.float32) / float(np.iinfo(w.dtype).max + 1)
        w = w.astype(np.float32)
        if w.ndim > 1:
            w = w.mean(1)
        kw = {"generate_kwargs": {"language": a.language}} if a.language else {}
        text = asr({"raw": w, "sampling_rate": sr}, return_timestamps=len(w) > 30 * sr, **kw)["text"]
        print(f"{p}: {text.strip()}", flush=True)


if __name__ == "__main__":
    main()
