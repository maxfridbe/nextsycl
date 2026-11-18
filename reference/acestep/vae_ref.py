"""The ACE-Step VAE's first windows on upstream's own class (diffusers AutoencoderOobleck, CPU, float32) against the
engine's vaebench outputs (a run's latents.f32 [frames, 64] and wav.f32 [2, n]; the source song at 48 kHz stereo).

    python -I vae_ref.py VAE_DIR SONG.wav RUN_DIR [RUN_DIR ...]

The engine's first encode window is [0, 28 s) (a 26 s core, 2 s of overlap after it): its 650 core frames are the mean
of upstream's encode of those 28 s, trimmed. Its first decode window (384 frames: a 256-frame core, 64 after) of the
latents is frames [0, 320): the first 256 x 1920 samples are upstream's decode of those frames, trimmed.
"""
import sys
import time
import wave

import numpy as np
import torch
from diffusers import AutoencoderOobleck

vae_dir, song, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
torch.set_grad_enabled(False)
vae = AutoencoderOobleck.from_pretrained(vae_dir, torch_dtype=torch.float32).eval()

w = wave.open(song)
assert w.getframerate() == 48000 and w.getnchannels() == 2 and w.getsampwidth() == 2
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2).T.astype(np.float32) / 32768.0
n_enc = 28 * 48000


def stats(got, want):
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    snr = 10 * np.log10((want.astype(np.float64) ** 2).sum() / max((d ** 2).sum(), 1e-300))
    return f"max |d| {d.max():.3e}  mean |d| {d.mean():.3e}  SNR {snr:.1f} dB"


t0 = time.time()
mean = vae.encode(torch.from_numpy(pcm[None, :, :n_enc].copy())).latent_dist.mean[0].numpy()  # [64, 700]
print(f"upstream encode of [0, 28 s): {mean.shape}, {time.time() - t0:.1f} s on the CPU")
core = mean[:, :650].T  # frames-major, as the engine's

decoded = {}
for r in runs:
    lat = np.fromfile(f"{r}/latents.f32", dtype="<f4").reshape(-1, 64)
    print(f"{r}: encode, first window's 650 core frames vs upstream: {stats(lat[:650], core)}")
    wav = np.fromfile(f"{r}/wav.f32", dtype="<f4").reshape(2, -1)
    # the decode's input: that run's own latents unless it decoded another run's (bench.json says nothing of that: the
    # against run's latents, when they are there, were used - the caller names the run whose latents were decoded first)
    key = r if len(decoded) == 0 else list(decoded)[0]
    if not decoded:
        z = torch.from_numpy(lat[:320].T.copy())[None]
        t0 = time.time()
        decoded[r] = vae.decode(z).sample[0].numpy()[:, : 256 * 1920]
        print(f"upstream decode of frames [0, 320) of {r}'s latents: {time.time() - t0:.1f} s on the CPU")
    print(f"{r}: decode, first window's core (256 frames) vs upstream on {key}'s latents: {stats(wav[:, : 256 * 1920], decoded[key])}")
