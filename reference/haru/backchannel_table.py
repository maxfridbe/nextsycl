"""Ours (runs/: over the socket; off/: offline through the build copy) against upstream's CPU runs (up/) on the same
user audio: per run, the share of the monologue it makes sound over (frames > -40 dBFS), its short stretches (at most
1.6 s - candidates for a back-channel; each transcribed once by Whisper, cached), and the back-channels among them
(every word a back-channel word). Per group (source x card): runs, the mean / median share, back-channels and the runs
with any.
    python3 -I backchannel_table.py ROOT
ROOT holds runs/, off/ and up/ (dirs backchannels.py has analysed) and tx.sh (a WAV -> Whisper's text).
"""
import json
import os
import re
import statistics
import subprocess
import sys

BC = {"yeah", "yes", "yep", "mm", "mhm", "mm-hmm", "mmhmm", "hmm", "uh-huh", "uh", "huh", "right", "okay", "ok", "oh", "wow", "really",
      "i", "see", "exactly", "sure", "true", "ah", "aha", "nice", "cool", "totally", "mm-hm", "hm", "um", "absolutely", "interesting", "no", "yup", "uh-hum"}


def words(t):
    return [w for w in re.split(r"[^a-z\-']+", t.lower()) if w]


def main(root):
    groups = {}
    detail = []
    for src in ("runs", "off", "up"):
        d = os.path.join(root, src)
        if not os.path.isdir(d):
            continue
        for r in sorted(os.listdir(d)):
            p = os.path.join(d, r, "bc.json")
            if not os.path.isfile(p):
                continue
            b = json.load(open(p))
            card = r.split("-")[3]
            bcs = []
            for s in b["stretches"]:
                if s["to_s"] - s["from_s"] > 1.6:
                    continue
                txt = os.path.join(d, r, s["file"].replace(".wav", ".txt"))
                if not os.path.isfile(txt):
                    out = subprocess.run([os.path.join(root, "tx.sh"), os.path.join(d, r, s["file"])], capture_output=True, text=True).stdout.strip()
                    open(txt, "w").write(out)
                w = words(open(txt).read())
                # a lone "no", "i", "um" or "uh" is not a back-channel
                if w and len(w) <= 3 and all(x in BC for x in w) and w not in (["no"], ["i"], ["um"], ["uh"]):
                    bcs.append((s["from_s"], s["over"], open(txt).read().strip()))
            share = b["loud_frames_in_monologue"] / b["frames_in_monologue"]
            g = groups.setdefault((src, card), {"runs": 0, "share": [], "bc": 0, "with_bc": 0, "silent": 0, "words": []})
            g["runs"] += 1
            g["share"].append(share)
            g["bc"] += len(bcs)
            g["with_bc"] += bool(bcs)
            g["silent"] += share < 0.02
            g["words"] += [x[2] for x in bcs]
            detail.append({"src": src, "run": r, "share": round(share, 3), "bc": bcs})
    json.dump({"detail": detail}, open(os.path.join(root, "compare.json"), "w"), indent=1)
    print(f"{'source':5} {'card':7} {'runs':>4} {'mean share':>10} {'median':>7} {'silent':>6} {'b-c':>4} {'runs w/ b-c':>11}  words")
    for (src, card), g in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"{src:5} {card:7} {g['runs']:4} {statistics.mean(g['share']):10.2f} {statistics.median(g['share']):7.2f} {g['silent']:6} {g['bc']:4} {g['with_bc']:11}  {g['words']}")


if __name__ == "__main__":
    main(sys.argv[1])
