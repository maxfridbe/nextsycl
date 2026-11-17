"""Our engine offline on the same inputs as the upstream CPU runs: the user side a socket run sent (its user.wav), a
card text, a voice, seeds - through the build copy's audio server (/v1/audio/speech: haru answers a recording as
haru-inference does, frame k + 1 from user frame k). Writes off/<run>-o<seed>/ like a socket run (model.wav, user.wav,
summary.json, an events.jsonl of audio frames only) for analyze.py.
    python3 -I offline_answer.py URL RUNS_DIR OUT_DIR VOICE.wav SEEDS RUN...
RUN names end in -<card>: OUT_DIR's parent holds card-<card>.txt (the rendered card).
"""
import base64
import json
import os
import shutil
import sys
import urllib.request
import wave


def main():
    url, runs, out, voice, seeds = sys.argv[1:6]
    vb = base64.b64encode(open(voice, "rb").read()).decode()
    for run in sys.argv[6:]:
        card = open(os.path.join(os.path.dirname(out.rstrip("/")) or ".", f"card-{run.split('-')[3]}.txt")).read()
        ub = base64.b64encode(open(os.path.join(runs, run, "user.wav"), "rb").read()).decode()
        for seed in [int(s) for s in seeds.split(",")]:
            d = os.path.join(out, f"{run}-o{seed}")
            if os.path.isfile(os.path.join(d, "model.wav")):
                continue
            os.makedirs(d, exist_ok=True)
            body = json.dumps({"instructions": card, "audio": ub, "ref_audio": vb, "seed": seed, "response_format": "wav"}).encode()
            r = urllib.request.urlopen(urllib.request.Request(url + "/v1/audio/speech", data=body, headers={"Content-Type": "application/json"}), timeout=600)
            open(os.path.join(d, "model.wav"), "wb").write(r.read())
            shutil.copy(os.path.join(runs, run, "user.wav"), os.path.join(d, "user.wav"))
            with wave.open(os.path.join(d, "model.wav")) as w:
                n = w.getnframes() // 3200
            with open(os.path.join(d, "events.jsonl"), "w") as f:
                for _ in range(n):
                    f.write('{"type": "response.audio.delta"}\n')
            s = json.load(open(os.path.join(runs, run, "summary.json")))
            s.update({"offline": True, "seed": seed})
            json.dump(s, open(os.path.join(d, "summary.json"), "w"), indent=1)
            print(d, n, "frames", flush=True)


if __name__ == "__main__":
    main()
