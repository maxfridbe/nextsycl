"""Upstream's own DuplexSession (github.com/higurashi-ai/Haru, haru/) on the CPU: the same user side a socket run sent
(its user.wav, frame by frame, as haru-inference does), the same card and voice, the release's sampling defaults
(temperature 0.7, top-k 25, DRY, guidance 1.5), a seed. Writes OUT/model.wav, OUT/user.wav, OUT/events.jsonl (the
text deltas and audio deltas in the socket's order, so backchannels.py reads it like a socket run) and OUT/summary.json.

    python -I upstream_session.py --src HARU/haru --model DIR --voice voice.wav --run SOCKET_RUN_DIR --card card.json
                              --seed S --out OUT [--dtype float32|bfloat16] [--threads 16]
"""
import argparse
import json
import os
import shutil
import sys
import time
import wave

import numpy as np


def read16(path):
    with wave.open(path) as w:
        assert w.getframerate() == 24000 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import torch
    torch.set_num_threads(a.threads)
    from haru.audio import prepare_reference
    from haru.models.loaders import load_model
    from haru.prompt import card_templates, render_card
    from haru.session import DuplexConfig, DuplexSession

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model = load_model(a.model, device="cpu", dtype=getattr(torch, a.dtype))
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    import importlib.resources as res
    preset = json.loads((res.files("haru") / "assets" / "cards" / "en.json").read_text())["chat"]
    preset.update(json.load(open(a.card)))
    card = render_card(preset, card_templates()["templates"]["en"])
    user = read16(os.path.join(a.run, "user.wav"))
    n = len(user) // 3200
    voice = prepare_reference(read16(a.voice))
    s = DuplexSession(model, card, voice, DuplexConfig(seed=a.seed, max_frames=n + 2))
    frames = [s.start()]
    print(f"prompt {frames[0].seconds:.1f} s, {s.prompt_length} positions", flush=True)
    for k in range(n):
        frames.append(s.step(user[k * 3200:(k + 1) * 3200]))
        if k % 50 == 0:
            print(f"frame {k}/{n} {frames[-1].seconds:.2f} s: {''.join(f.text for f in frames)[-80:]!r}", flush=True)
    model_audio = np.concatenate([f.audio for f in frames])
    with wave.open(os.path.join(a.out, "model.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((np.clip(model_audio, -1, 1) * 32767).astype("<i2").tobytes())
    shutil.copy(os.path.join(a.run, "user.wav"), os.path.join(a.out, "user.wav"))
    with open(os.path.join(a.out, "events.jsonl"), "w") as f:
        for fr in frames:
            if fr.text:
                f.write(json.dumps({"type": "response.audio_transcript.delta", "delta": fr.text}) + "\n")
            f.write(json.dumps({"type": "response.audio.delta"}) + "\n")
    summ = json.load(open(os.path.join(a.run, "summary.json")))
    summ.update({"upstream": True, "seed": a.seed, "dtype": a.dtype, "model_text": " ".join("".join(f.text for f in frames).split()),
                 "tokens": [f.token for f in frames], "card": card})
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(summ["model_text"])


if __name__ == "__main__":
    main()
