"""Back-channel analysis of one reference/haru/client.py run, or an upstream_session.py / offline_answer.py one (DIR:
model.wav, user.wav, events.jsonl, summary.json).

Frame j of model.wav plays beside user frame j (client.py's call.wav alignment). For each model frame: its level
(dBFS RMS) and its text (the transcript delta sent just before its audio delta). A model frame is "sound" above
-40 dBFS. Stretches of sound inside the monologue's span (the --say clip, from_s..to_s) are cut out (padded 0.2 s)
to DIR/bc-NN.wav for Whisper; each is tagged by whether the user's own frames beside it are voiced (> -40 dBFS) or
a pause in the monologue. Writes DIR/bc.json.
    python -I backchannels.py DIR   (numpy)
"""
import json
import os
import sys
import wave

import numpy as np

FRAME = 3200
RATE = 24000
THR = -40.0


def read(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)


def main(d):
    model = read(os.path.join(d, "model.wav"))
    user = read(os.path.join(d, "user.wav"))
    summ = json.load(open(os.path.join(d, "summary.json")))
    nf = min(len(model), len(user)) // FRAME
    mdb = np.array([db(model[i * FRAME:(i + 1) * FRAME]) for i in range(nf)])
    udb = np.array([db(user[i * FRAME:(i + 1) * FRAME]) for i in range(nf)])
    # text per frame
    text = [""] * (nf + 2)
    n_audio = 0
    for l in open(os.path.join(d, "events.jsonl")):
        v = json.loads(l)
        t = v.get("type", "")
        if t.endswith("audio.delta"):
            n_audio += 1
        elif t.endswith("audio_transcript.delta") and n_audio < len(text):
            text[n_audio] += v.get("delta", "")
    say = summ["user_said"][0]
    a, b = int(round(say["from_s"] * RATE / FRAME)), int(round(say["to_s"] * RATE / FRAME))
    loud = mdb > THR
    # stretches of loud model frames inside [a, b), gaps of up to 2 frames merged
    st = []
    i = a
    while i < b:
        if loud[i]:
            j = i
            while j < b and (loud[j] or (j + 1 < b and loud[j + 1]) or (j + 2 < b and loud[j + 2])):
                j += 1
            while not loud[j - 1]:
                j -= 1
            st.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for k, (i, j) in enumerate(st):
        lo, hi = max(0, i * FRAME - int(0.2 * RATE)), min(len(model), j * FRAME + int(0.2 * RATE))
        p = os.path.join(d, f"bc-{k:02d}.wav")
        with wave.open(p, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((np.clip(model[lo:hi], -1, 1) * 32767).astype("<i2").tobytes())
        voiced = int(np.sum(udb[i:j] > THR))
        out.append({"file": os.path.basename(p), "from_s": round(i * FRAME / RATE, 2), "to_s": round(j * FRAME / RATE, 2),
                    "frames": j - i, "peak_dbfs": round(float(mdb[i:j].max()), 1),
                    "user_voiced_frames_beside": voiced, "over": "speech" if voiced * 2 >= (j - i) else "pause",
                    "model_text": "".join(text[i:j + 1]).strip()})
    after = loud[b:].sum() if b < nf else 0
    res = {"dir": d, "clip": say, "frames_in_monologue": b - a, "loud_frames_in_monologue": int(loud[a:b].sum()),
           "loud_frames_over_user_voiced": int(np.sum(loud[a:b] & (udb[a:b] > THR))),
           "user_voiced_frames": int(np.sum(udb[a:b] > THR)), "loud_frames_before": int(loud[:a].sum()),
           "loud_frames_after": int(after), "frames_after": int(max(0, nf - b)),
           "first_loud_after_end_s": next((round((k - b) * FRAME / RATE, 2) for k in range(b, nf) if loud[k]), None),
           "text_before": "".join(text[:a]).strip(), "text_during": " ".join("".join(text[a:b + 1]).split()),
           "text_after": " ".join("".join(text[b + 1:]).split()), "stretches": out}
    json.dump(res, open(os.path.join(d, "bc.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "dir"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
