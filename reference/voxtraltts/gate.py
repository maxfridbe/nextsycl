"""The encoderless clone's gate (docs/proposals/voxtral-encoderless-clone.md): speaker likeness and word errors of
speech against a reference recording, on the CPU.

    python gate.py likeness REF.wav A.wav [B.wav ...]          cosine of speaker embeddings, REF against each
    python gate.py words LANG "the text" A.wav [B.wav ...]     Whisper small's transcript and its WER against the text
    python gate.py table PLAN.json OUT.json                    both for a plan: {"refs": {name: {"wav", "text", "lang"}},
                                                               "runs": [{"ref", "system", "wav", "text"}]}

Likeness is microsoft/wavlm-base-plus-sv's x-vector (a speaker-verification head on WavLM; cosine above ~0.86 is its
same-speaker threshold); words are openai/whisper-small's transcript, both lower-cased with punctuation dropped before
the word error rate.
"""
import json
import re
import sys
import unicodedata

import librosa
import numpy as np
import torch

torch.set_num_threads(8)
_sv = None
_emb = {}
_asr = None


def wav16(path):
    import soundfile as sf

    w, sr = sf.read(path, dtype="float32", always_2d=True)
    return librosa.resample(w.mean(1), orig_sr=sr, target_sr=16000)


def embed(path):
    global _sv
    if path in _emb:
        return _emb[path]
    from transformers import AutoFeatureExtractor, WavLMForXVector

    if _sv is None:
        _sv = (AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"), WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval())
    fe, m = _sv
    x = fe(wav16(path), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        e = m(**x).embeddings[0]
    _emb[path] = torch.nn.functional.normalize(e, dim=-1)
    return _emb[path]


def likeness(ref, other):
    return float(embed(ref) @ embed(other))


_said = {}


def transcribe(path, lang):
    global _asr
    if (path, lang) in _said:
        return _said[(path, lang)]
    from transformers import pipeline

    if _asr is None:
        _asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu", torch_dtype=torch.float32, chunk_length_s=30)
    t = _asr({"raw": wav16(path), "sampling_rate": 16000}, generate_kwargs={"language": lang})["text"].strip()
    _said[(path, lang)] = t
    return t


def norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    return [w for w in re.sub(r"[^\w\s']", " ", t).split() if w]


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d.copy(), i
        for j, hw in enumerate(h, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (rw != hw))
    return d[len(h)] / max(1, len(r))


def main():
    cmd = sys.argv[1]
    if cmd == "likeness":
        for p in sys.argv[3:]:
            print(f"{likeness(sys.argv[2], p):.4f} {p}", flush=True)
    elif cmd == "words":
        lang, text = sys.argv[2], sys.argv[3]
        for p in sys.argv[4:]:
            t = transcribe(p, lang)
            print(f"{wer(text, t):.3f} {p} | {t}", flush=True)
    elif cmd == "table":
        plan = json.load(open(sys.argv[2]))
        out = []
        for r in plan["runs"]:
            ref = plan["refs"][r["ref"]]
            t = transcribe(r["wav"], ref.get("lang", "english"))
            row = {**r, "likeness": likeness(ref["wav"], r["wav"]), "transcript": t, "wer": wer(r["text"], t)}
            out.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        json.dump(out, open(sys.argv[3], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
