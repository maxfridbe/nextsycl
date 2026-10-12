#!/usr/bin/env python3
"""Reference test: a short dialogue scene with cuts between people, made through the front door's API alone.

Two people talk across hard cuts - the captain, then the engineer, the captain, the engineer - in one place seen
from two sides, between an establishing shot and one of her lifting a lantern and walking away. It is the recipe
for people talking (docs/video/scenes.md, S2, S3, S13):

- the pictures are **references, not composites**: a background (two views of one place, no people), a character sheet
  for each person (front, three-quarter, side and a smiling close-up on a plain ground) and a prop. Each clip is Ref2VA
  with the person's sheet, their place and their voice (`ref_images`, `ref_audios`), and the prompt says "the woman
  from the character sheet <Picture 1> stands here, on the quay of <Picture 2>". (`image.edit` of a person into a
  place returned the place without them.) Nothing is pinned: the shot is held by the words;
- the voice is one steady 3 s+ sample per person on every clip, the first too (a clip made without it gets whatever
  voice the model draws, a man's for her, and a cut to one with it changes voice);
- one speaker in each shot (two voices in one shot put a voice on the wrong mouth);
- warm: golden-hour light, warm clothes, a smile in the prompt and in the voice sample's instruction - a violet dusk
  and dark coats gave cold, flat people;
- each clip is sized to its line (words / 3.3 / 0.57 s, so a line is not rushed), cut to the next with a plain cut
  (nothing is carried, so nothing overlaps) and each clip's sound levelled alone (one voice, one room).

Then it checks what it made - the clips' sizes, lengths and sound, and that each person's clips hold that person's
voice (median pitch within 25% of the reference's, and the captain's clips higher than the engineer's) - and joins
them here with ffmpeg.

    tests/api/harbour_dialogue.py                       # 768x576, 12 steps, upscaled 1.5: ~25 min
    tests/api/harbour_dialogue.py --quick               # 384x288, 8 steps, no upscale: ~6 min
    tests/api/harbour_dialogue.py --url http://box:8000 --token ns-...

Needs: Python 3.9+, an ffmpeg and ffprobe with H.264; on the server minimax-h3 (INT8) and minimax-h3-ref2va, an image
model (qwen-image-2.1-q8) and a speech model (qwen3-tts-custom). Writes into --out: pictures, voices, clips, a contact
sheet per clip, film.mp4 and report.json. Exits 1 when a check fails.
"""

import argparse
import base64
import json
import shlex
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaiju_lighthouse as K   # the API client, the probes, the polling

FPS = 24
STYLE = ("Live-action, cinematic, warm golden-hour sunlight, soft amber and honey tones, natural warm skin tones, "
         "shallow depth of field.")
MARA = ("a woman in her forties with short grey-streaked black hair, grey eyes and a kind, weathered face, in a mustard-yellow "
        "oilskin coat over a cream wool sweater and a rust-red scarf")
ILYA = ("an old man with a full white beard, kind blue eyes and a ruddy face, in a flat tweed cap and a rust-brown wool jersey with "
        "rolled sleeves")
PLACE = ("a stone harbour quay with iron bollards in the golden hour, fishing boats in red and blue at their moorings, honey-coloured "
         "stone houses on the hill, calm water glittering with warm light")
SHEET = ("character reference sheet, photographic: full-body front view, three-quarter view and side view, and a close-up of the "
         "smiling face, on a plain light-grey studio background, even soft warm light")
# The pictures are references, not composites: a place (two views of it, no people), a character sheet for each person and
# a prop. The video model puts "this character standing here" together (Ref2VA: `<Picture i>` in the prompt) - image.edit
# of a person into a place returned the place without the person, or kept the old room behind them and over-processed it.
PICTURES = {   # name -> (prompt, size)
    "harbour": (f"wide cinematic photograph of {PLACE}, no people", "1024x768"),
    "place_a": (f"cinematic photograph of {PLACE}, looking out over the water, an iron bollard in the foreground, the boats and the far "
                "shore beyond, no people", "1024x768"),
    "place_b": (f"cinematic photograph of {PLACE}, looking back along the quay: a weathered stone boathouse wall and the honey-coloured "
                "houses behind, an iron bollard and cobbles in the foreground, no people", "1024x768"),
    "sheet_mara": (f"{SHEET}: {MARA}", "1344x768"),
    "sheet_ilya": (f"{SHEET}: {ILYA}", "1344x768"),
    "lantern": ("photograph of a brass ship's lantern with a warm glowing flame and a ring handle, on a plain dark background", "1024x768"),
}
PLACE_OF = {"mara": "place_a", "ilya": "place_b"}   # where each person stands: the two views of the quay
WARM = "warm, kind and friendly, a gentle smile in the voice"
# (who, the line): the voice reference is the speech model's rendering of a line of the same person's
VOICE = {"mara": ("vivian", "We sail at first light, my friend, and I am glad you are here."),
         "ilya": ("ryan", "She will hold, captain, she always does for you.")}
DESIGN = {"mara": "a woman in her forties, a warm low alto, calm and weathered, speaking slowly and steadily like a ship's captain",
          "ilya": "an old man, a gravelly deep baritone, dry and unhurried, with the tired humour of an engineer"}
SCENE = [   # (id, who, line) - who None: no one speaks
    ("establish", None, None),
    ("mara_1", "mara", "We sail at first light, my friend, and I am glad you are here."),
    ("ilya_1", "ilya", "She will hold, captain, she always does for you."),
    ("mara_2", "mara", "Thank you, Ilya. Now get some rest before the dawn."),
    ("ilya_2", "ilya", "Aye, captain. Go well, and bring the lantern back."),
    ("walk", None, None),
]
LOCK = ("The camera sits in one fixed position on a tripod; that exact shot size, light level and colour hold from the first "
        "frame to the last.")


def snap(frames):
    """A clip's frame count as the engine makes it: 17k + 5, up from the count"""
    n = max(5, round(frames))
    return n + (5 - n) % 17


def frames_for(line):
    """A line's clip: words / 3.3 / 0.57 seconds (the model speaks for 57% of a clip, 3.3 words a second), at least ~4.5 s"""
    return snap(max(len(line.split()) / 3.3 / 0.57 + 0.5, 4.4) * FPS) if line else snap(4.4 * FPS)


def prompt_of(cid, who, line):
    if cid == "establish":
        return (f"integrated_multimodal_description: [Shot 1] {STYLE} A wide establishing shot of a small harbour in the golden hour: "
                "red and blue fishing boats at their moorings, a stone quay, honey-coloured houses on the hill, the water glittering "
                "with warm light. The camera is static.\n\noverall_soundscape: Water lapping against stone, a distant gull, a rope "
                "creaking.\n\nnon_diegetic_music: None.")
    if cid == "walk":
        return (f"integrated_multimodal_description: [Shot 1] {STYLE} A medium shot: the woman from the character sheet <Picture 1> "
                "stands here on the quay of <Picture 2>, smiles, bends and lifts the brass lantern <Picture 3> by its ring, and walks "
                "away along the quay toward the moored boats, the golden light warm on her coat. The camera tracks behind her at a "
                "steady distance.\n\noverall_soundscape: Footsteps on stone, water lapping softly, a distant gull.\n\n"
                "non_diegetic_music: None.")
    person = "the woman" if who == "mara" else "the old man"
    pron = "her" if who == "mara" else "his"
    room = "Her warm voice, close and clear; water lapping softly against stone, a distant gull." if who == "mara" else \
        "His warm voice, close and clear; water lapping softly against stone, a distant gull."
    return (f"integrated_multimodal_description: [Shot 1] {STYLE} A medium close-up, head and shoulders: {person} from the character "
            f"sheet <Picture 1> stands here, on the quay of <Picture 2>, the golden light warm on {pron} face. {person.capitalize()} "
            f"smiles gently, blinks, looks straight into the camera, then speaks in the voice of <Audio 1> (S1): <d>[English] {line}</d> "
            f"The lips move with the words, the eyes stay warm and the smile returns at the end. {LOCK}\n\n"
            f"overall_soundscape: {room}\n\nnon_diegetic_music: None.")


def pcm(f, rate):
    b = subprocess.run([*K.FFMPEG, "-v", "error", "-i", str(f), "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
                       capture_output=True).stdout
    return list(struct.unpack(f"<{len(b) // 2}h", b))


def pitch(f, rate=8000):
    """The median pitch (Hz) of a voice's louder, voiced frames, by autocorrelation; None when nothing is voiced. Octave
    errors are possible - enough to tell a woman's voice from a man's, not a speaker from a speaker."""
    x = pcm(f, rate)
    n, h = int(0.04 * rate), int(0.02 * rate)
    lo, hi = int(rate / 400), int(rate / 70)
    starts = list(range(0, len(x) - n - hi, h))
    en = [sum(v * v for v in x[i:i + n]) for i in starts]
    if not en:
        return None
    thr = sorted(en)[int(len(en) * 0.6)]
    out = []
    for i, e in zip(starts, en):
        if e < thr:
            continue
        r0 = sum(v * v for v in x[i:i + n])
        best, bl = 0, 0
        for lag in range(lo, hi):
            c = sum(x[i + j] * x[i + j + lag] for j in range(n))
            if c > best:
                best, bl = c, lag
        if bl and best > 0.5 * r0:
            out.append(rate / bl)
    return round(statistics.median(out)) if len(out) >= 10 else None


def sheet(f, dst, n=6, size=160):
    subprocess.run([*K.FFMPEG, "-y", "-v", "error", "-i", str(f), "-vf",
                    f"select='not(mod(n\\,{max(1, K.probe(f)['frames'] // n)}))',scale={size}:-1,tile={n}x1", "-frames:v", "1", str(dst)], check=True)


def cut_join(clips, targets, out, fade=1.0):
    """Plain cuts: each clip's sound levelled alone to its LUFS (one voice, one room), 6 ms in and out so a cut does not
    click, the picture untouched; the film's last `fade` seconds faded out"""
    n, parts = len(clips), []
    for i, (c, t) in enumerate(zip(clips, targets)):
        e = K.loudness(c)
        g = f"volume={t - e:.2f}dB," if e is not None else ""
        d = K.probe(c)["frames"] / FPS
        # the sound exactly the picture's length: a clip's sound runs a fraction of a frame long, and concat would hold a frame
        parts.append(f"[{i}:a]{g}apad,atrim=end={d:.6f},asetpts=PTS-STARTPTS,alimiter=limit=0.891:level=disabled,"
                     f"afade=t=in:d=0.006,afade=t=out:st={d - 0.006:.4f}:d=0.006[a{i}]")
    fc = parts + ["".join(f"[{i}:v][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"]
    total = sum(K.probe(c)["frames"] for c in clips) / FPS
    fc.append(f"[v]fade=t=out:st={total - fade:.3f}:d={fade}[vf];[a]afade=t=out:st={total - fade:.3f}:d={fade}[af]")
    cmd = [*K.FFMPEG, "-y", "-v", "error"]
    for c in clips:
        cmd += ["-i", str(c)]
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vf]", "-map", "[af]", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token")
    ap.add_argument("--out", default="harbour-dialogue")
    ap.add_argument("--quick", action="store_true", help="384x288, 8 steps, no upscale")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--upscale", type=float)
    ap.add_argument("--seed", type=int, default=1954)
    ap.add_argument("--image-model", default="qwen-image-2.1-q8")
    ap.add_argument("--speech-model", default="qwen3-tts-custom")
    ap.add_argument("--design-model", default="qwen3-tts-design")
    ap.add_argument("--voices", choices=["sample", "lines", "design"], default="sample",
                    help="the voice references: one sample a person (built-in voices), each clip's own line spoken by the speech "
                         "model, or one sample a person in a voice the design model made up")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    a = ap.parse_args()
    K.FFMPEG, K.FFPROBE = shlex.split(a.ffmpeg), shlex.split(a.ffprobe)
    K.ffmpeg_can()
    w, h = a.width or (384 if a.quick else 768), a.height or (288 if a.quick else 576)
    steps = a.steps or (8 if a.quick else 12)
    upscale = a.upscale if a.upscale is not None else (1.0 if a.quick else 1.5)

    api = K.Api(a.url, K.token_of(a.token))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = time.strftime("%Y%m%d-%H%M%S")
    project = f"harbour-dialogue {run}"
    report = {"project": project, "settings": {"voices": a.voices, "width": w, "height": h, "steps": steps, "upscale": upscale, "seed": a.seed}, "clips": {}, "checks": {}}
    files = {}

    def up(name, path):
        files[Path(name).stem] = api.rpc("video.upload", {"name": name, "data": base64.b64encode(Path(path).read_bytes()).decode()})["name"]

    # 1. pictures and voices (the video models off the card first)
    for m in ("minimax-h3", "minimax-h3-q6k", "minimax-h3-q4km", "minimax-h3-ref2va"):
        try:
            api.rpc("system.unload", {"model": m})
        except RuntimeError:
            pass
    for p, (prompt, size) in PICTURES.items():
        dst = out / f"{p}.png"
        if not dst.exists():
            print(f"picture {p}: {a.image_model} {size}", flush=True)
            r = api.rpc("image.generate", {"model": a.image_model, "prompt": prompt, "size": size, "steps": 28, "seed": a.seed})
            url = r["data"][0]["url"]
            api.get(url[url.index("/v1/"):], dst)
        up(f"{p}.png", dst)
    api.rpc("system.unload", {"model": a.image_model})
    # the voice references, three ways: `sample` one recording a person (the speech model, a built-in voice, a stock line),
    # `lines` every clip's own line spoken by the speech model in that person's voice, `design` one recording a person
    # in a voice the design model made up from a description
    model = a.design_model if a.voices == "design" else a.speech_model
    wanted = ([(f"line_{cid}", who, line) for cid, who, line in SCENE if who] if a.voices == "lines"
              else [(f"voice_{who}", who, VOICE[who][1]) for who in VOICE])
    pitches = {"mara": [], "ilya": []}
    for name, who, text in wanted:
        dst = out / f"{name}.wav"
        if not dst.exists():
            print(f"voice {name}: {model}", flush=True)
            body = {"model": model, "input": text, "language": "english", "response_format": "url"}
            if a.voices == "design":
                body["instructions"] = DESIGN[who]
            else:
                body["voice"] = VOICE[who][0]
                body["instructions"] = WARM
            r = api.rpc("audio.speak", body)
            url = r.get("url") or r["data"][0]["url"]
            api.get(url[url.index("/v1/"):], dst)
        up(f"{name}.wav", dst)
        pitches[who].append(pitch(dst, 16000))
    api.rpc("system.unload", {"model": model})
    refs_hz = {who: (round(statistics.median([x for x in v if x])) if any(v) else None) for who, v in pitches.items()}
    report["voices_hz"] = refs_hz
    print(f"voices: pitch {refs_hz}", flush=True)

    # 2. the clips, in the scene's order but by engine (INT8 first: the establishing shot)
    t0 = time.time()
    names, local = {}, []
    for i, (cid, who, line) in enumerate(SCENE, 1):
        nf = frames_for(line)
        body = {"prompt": prompt_of(cid, who, line), "seconds": round(nf / FPS, 4), "steps": steps, "seed": a.seed + i, "width": w, "height": h,
                "project": project, "queue": True, "label": f"Dialogue {run} {i:02}/{len(SCENE):02}: {cid}"}
        if cid == "establish":
            body |= {"engine": "INT8", "first_frame": files["harbour"]}   # the quay, wide
        elif cid == "walk":
            body |= {"engine": "REF2VA", "ref_images": [files["sheet_mara"], files["place_a"], files["lantern"]]}
        else:
            body |= {"engine": "REF2VA", "ref_images": [files[f"sheet_{who}"], files[PLACE_OF[who]]],
                     "ref_audios": [files[f"line_{cid}" if a.voices == "lines" else f"voice_{who}"]]}
        if upscale > 1:
            body |= {"upscale": upscale, "upscaler": "esrgan-general"}
        t = time.time()
        api.rpc("video.generate", body)
        print(f"{body['label']}: queued ({nf} frames{', ' + str(len(line.split())) + ' words' if line else ''})", flush=True)
        name = K.wait_clip(api, project, i, body["label"])
        dst = api.get(f"/v1/video/files/{name}", out / f"{i:02}_{cid}.mp4")
        p = K.probe(dst)
        sheet(dst, out / f"{i:02}_{cid}.png")
        row = {"file": name, "frames": p["frames"], "asked": nf, "width": p["width"], "height": p["height"], "audio": p["audio"],
               "wall_s": round(time.time() - t), "pitch_hz": pitch(dst) if who else None}
        report["clips"][cid] = row
        print(f"{cid}: {name} in {row['wall_s']} s, {p['width']}x{p['height']}, {p['seconds']:.2f} s"
              f"{', pitch ' + str(row['pitch_hz']) + ' Hz' if who else ''}", flush=True)
        names[cid] = name
        local.append(dst)

    # 3. the film: plain cuts, each clip's sound levelled alone (the establishing shot and the walk under the voices)
    film = out / "film.mp4"
    targets = [-24.0 if who is None else -18.0 for _, who, _ in SCENE]
    total = cut_join(local, targets, film)
    fp = K.probe(film)

    # 4. checks
    ck = report["checks"]
    ck["clips are the size asked"] = all(c["width"] >= w and c["height"] >= h for c in report["clips"].values())
    ck["clips are the length asked"] = all(c["frames"] == c["asked"] for c in report["clips"].values())
    ck["clips have sound"] = all(c["audio"] for c in report["clips"].values())
    ck["film is the clips cut together"] = abs(fp["frames"] - sum(c["frames"] for c in report["clips"].values())) <= 1 and fp["audio"]
    for who in ("mara", "ilya"):
        mine = [c["pitch_hz"] for cid, c in report["clips"].items() if cid.startswith(who)]
        ref = refs_hz.get(who)
        ck[f"{who}'s voice held in every clip (within 25% of her reference {ref} Hz)"] = bool(ref) and all(
            m is not None and abs(m - ref) / ref <= 0.25 for m in mine)
    m = [c["pitch_hz"] for cid, c in report["clips"].items() if cid.startswith("mara") and c["pitch_hz"]]
    i = [c["pitch_hz"] for cid, c in report["clips"].items() if cid.startswith("ilya") and c["pitch_hz"]]
    ck["the captain's clips are higher than the engineer's"] = bool(m and i) and min(m) > max(i)
    report["film"] = {"file": str(film), "seconds": fp["seconds"], "wall_s": round(time.time() - t0)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    for k, v in ck.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    print(f"film: {film}, {fp['width']}x{fp['height']}, {fp['seconds']:.2f} s; report {out / 'report.json'}")
    return 0 if all(ck.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
