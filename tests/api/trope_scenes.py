#!/usr/bin/env python3
"""Reference test: two film tropes scripted over three clips each, made through the front door's API alone and judged for
consistency across the dimensions docs/video/tropes.md names.

    T14  the mystery object   - a woman finds a music box in an attic (medium), the box opens (extreme close-up), she
                                listens (medium close-up): the face, the box, the room and the light must be the same
                                three times while the shot size changes on purpose
    T30  farewell at the station - she says goodbye (medium close-up), he answers from the carriage door (the reverse),
                                the train leaves with her alone on the platform (wide): two faces and two voices, one
                                station, and the two people looking at each other across the cut

Made as harbour_dialogue.py makes a scene (docs/video/scenes.md S2, S3, S13): pictures are references, not composites - a
place, a character sheet for each person, a prop - and every clip is Ref2VA with the pictures it needs and, for a person
who speaks, that person's voice sample; one speaker a shot; warm light; each line's clip sized to the line.

Then it scores each trope, per dimension:
- measured: the clips' sizes, lengths and sound; for a speaker the median pitch against the voice sample's (as
  harbour_dialogue.py); that the film is the clips cut together;
- judged: a vision model (the chat model, default qwen3.8-27b-q4, temperature 0) is shown frames of the clips - and the
  reference pictures - and asked one question a dimension (the same person? the same box? the same room? the same
  light? did the shot size go as scripted? did they face each other?) for a score from 1 to 5; 4 or more passes. The
  judge is a model, not an oracle: every clip's contact sheet and the frames it saw are written beside the report, so a
  score can be argued with.

    tests/api/trope_scenes.py                         # both tropes, 768x576, 12 steps, upscaled 1.5: ~1 h
    tests/api/trope_scenes.py --only T14 --quick      # one, 384x288, 8 steps, no upscale: ~10 min
    tests/api/trope_scenes.py --no-judge              # the measured checks and the sheets only
    tests/api/trope_scenes.py --url http://box:8000 --token ns-...

Needs: Python 3.9+, an ffmpeg and ffprobe with H.264; on the server minimax-h3 (INT8) and minimax-h3-ref2va, an image model
(qwen-image-2.1-q8), a speech model (qwen3-tts-custom) and, for the judge, a vision chat model (qwen3.8-27b-q4). Writes into
--out/<trope>: pictures, voices, clips, a contact sheet and key frames per clip, film.mp4 and report.json. Exits 1 when a
check fails.
"""

import argparse
import base64
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaiju_lighthouse as K   # the API client, the probes, the polling
import harbour_dialogue as H   # the speech, pitch, sheet and cut helpers, and the harbour's people

FPS = 24
WARM_LIGHT = "Live-action, cinematic, warm golden light with soft amber tones, natural warm skin tones, shallow depth of field."
LOCK = "The camera sits in one fixed position on a tripod; that shot size, light level and colour hold from the first frame to the last."
# the same two people as the harbour, in a cardigan and a travelling coat (a different place, so a different costume)
MARA = ("a woman in her forties with short grey-streaked black hair, grey eyes and a kind, weathered face, in a mustard-yellow wool "
        "cardigan over a cream blouse and a rust-red scarf")
ILYA = ("an old man with a full white beard, kind blue eyes and a ruddy face, in a flat tweed cap and a rust-brown wool travelling "
        "coat with a worn leather satchel")
SHEET = H.SHEET


def shot(action, sound, music="None."):
    return (f"integrated_multimodal_description: [Shot 1] {WARM_LIGHT} {action}\n\noverall_soundscape: {sound}\n\n"
            f"non_diegetic_music: {music}")


# ---- T14: the mystery object -------------------------------------------------------------------------------------------
BOX = "a small antique music box of dark walnut with a mother-of-pearl star inlaid in the lid and brass hinges"
T14 = {
    "title": "the mystery object",
    "pictures": {
        "attic": ("cinematic photograph of a warm dusty attic with a round window, shafts of golden light with floating dust, an open "
                  "wooden trunk and old quilts, honey-coloured beams, no people", "1024x768"),
        "musicbox": (f"photograph of {BOX}, the lid closed, on a plain dark background, even soft warm light", "1024x768"),
        "sheet_mara": (f"{SHEET}: {MARA}", "1344x768"),
    },
    "voices": [],
    "clips": [
        # (id, engine, ref pictures, seconds, prompt)
        ("find", "REF2VA", ["sheet_mara", "attic", "musicbox"], 5.0,
         shot("A medium shot: the woman from the character sheet <Picture 1> kneels in the attic of <Picture 2> beside the open trunk, "
              "lifts out the music box <Picture 3> with both hands and holds it up in the golden light, studying it with curiosity. "
              f"{LOCK}", "Old wood creaking, a soft hush of dust, her breath.")),
        ("open", "REF2VA", ["musicbox", "attic"], 5.0,
         shot("An extreme close-up, filling the frame: the music box <Picture 1> held in two hands in the golden light of <Picture 2>; "
              "the hands lift its lid, the mother-of-pearl star inlay gleams, the tiny brass comb and cylinder begin to turn. "
              f"{LOCK}", "A delicate, slightly out-of-tune music-box melody begins, close and clear, with the small ticking of its gears.")),
        ("listen", "REF2VA", ["sheet_mara", "attic", "musicbox"], 5.0,
         shot("A medium close-up, head and shoulders: the woman from the character sheet <Picture 1> in the attic of <Picture 2> holds "
              "the open music box <Picture 3> close to her chest and listens, her eyes softening and glistening, a small, moved smile. "
              f"{LOCK}", "The same delicate music-box melody continues softly, her slow breath, old wood.")),
    ],
    # (dimension, question, [(clip id or picture, fraction of the clip)]) - the judge scores each from 1 to 5
    "judge": [
        ("D1 identity: the same woman", "Image 1 and image 2 are two shots of a film. Is the woman in image 2 the same person as in image 1 "
         "(same face, hair, clothes)?", [("find", 0.5), ("listen", 0.5)]),
        ("D2 prop: the same music box", "Image 1 is a reference picture of a music box. Images 2, 3 and 4 are shots of a film. Is the music "
         "box in each of images 2, 3 and 4 the same object as in image 1 (same dark wood, same star inlay in the lid, same brass hinges)? "
         "If it cannot be seen in one of them, give 1.", [("@musicbox", 0), ("find", 0.85), ("open", 0.6), ("listen", 0.5)]),
        ("D4 place: the same attic", "Are image 1 and image 2 shots in the same attic (the same beams, light and trunk)?",
         [("find", 0.5), ("listen", 0.5)]),
        ("D5 look: the same light", "Do images 1, 2 and 3 share one lighting and colour palette (warm golden light)? Give 1 if one of them "
         "looks cold or differently graded.", [("find", 0.5), ("open", 0.5), ("listen", 0.5)]),
        ("D8 scale: wide, then macro, then close", "The script says: image 1 a medium shot, image 2 an extreme close-up of the music box, "
         "image 3 a medium close-up of her face. Does the shot size go that way?", [("find", 0.5), ("open", 0.5), ("listen", 0.5)]),
        ("D10 performance: she is moved", "Image 1 and image 2 are the start and end of one shot. Does the woman's expression move from "
         "attentive to moved (softer, eyes glistening)?", [("listen", 0.1), ("listen", 0.9)]),
    ],
}

# ---- T30: farewell at the station ---------------------------------------------------------------------------------------
STATION = ("a small old railway station platform in the golden hour, a green steam train waiting at the platform, iron lamp posts "
           "with warm lamps, honey-coloured brick, wisps of steam")
T30 = {
    "title": "farewell at the station",
    "pictures": {
        "platform_a": (f"cinematic photograph of {STATION}, looking along the platform toward the train, no people", "1024x768"),
        "platform_b": (f"cinematic photograph of {STATION}, seen from the train's side: an open carriage door with a brass handrail and "
                       "the platform and lamp posts beyond, no people", "1024x768"),
        "train": ("photograph of a vintage green steam locomotive with brass fittings and red wheels at a station platform, three-quarter "
                  "view, golden light, no people", "1024x768"),
        "sheet_mara": (f"{SHEET}: {MARA}", "1344x768"),
        "sheet_ilya": (f"{SHEET}: {ILYA}", "1344x768"),
    },
    "voices": ["mara", "ilya"],
    "clips": [
        ("she", "REF2VA", ["sheet_mara", "platform_a", "@voice_mara"], None,
         ("Write to me when you arrive, and keep warm.", "mara"),
         lambda line: shot(
             "A medium close-up, head and shoulders: the woman from the character sheet <Picture 1> stands on the platform of <Picture 2>, "
             "turned toward the right of the frame where the train waits, the warm lamp light on her face, steam drifting behind her. She "
             f"looks up toward the right with glistening eyes, then speaks in the voice of <Audio 1> (S1): <d>[English] {line}</d> The lips "
             f"move with the words and she smiles bravely at the end. {LOCK}",
             "Her warm voice, close and clear; a locomotive idling, a soft hiss of steam, a distant station bell.")),
        ("he", "REF2VA", ["sheet_ilya", "platform_b", "@voice_ilya"], None,
         ("I will, and you will wait for me at the harbour.", "ilya"),
         lambda line: shot(
             "A medium close-up, head and shoulders, the reverse shot: the old man from the character sheet <Picture 1> stands in the open "
             "carriage doorway of <Picture 2>, holding the brass handrail, turned toward the left of the frame where she stands on the "
             f"platform, the warm lamp light on his face. He looks down toward the left with kind eyes, then speaks in the voice of <Audio 1> "
             f"(S1): <d>[English] {line}</d> The lips move with the words and he nods and smiles at the end. {LOCK}",
             "His warm voice, close and clear; a locomotive idling, a soft hiss of steam, a distant station bell.")),
        ("away", "REF2VA", ["sheet_mara", "platform_a", "train"], 5.0, (None, None),
         lambda line: shot(
             "A wide shot: the green steam locomotive <Picture 3> pulls slowly away along the platform of <Picture 2> toward the right, "
             "steam billowing; the woman from the character sheet <Picture 1> stands alone on the platform in the golden light, turned "
             "toward the train, lifting one hand in a slow wave as it goes. The camera is static.",
             "The locomotive's slow chuffing, steam, the clack of wheels, a distant whistle.")),
    ],
    "judge": [
        ("D1 identity: her", "Image 1 is a character sheet of a woman; images 2 and 3 are two shots of a film. Is the woman in each of "
         "images 2 and 3 the same person as on the sheet (same face, hair, clothes)?", [("@sheet_mara", 0), ("she", 0.5), ("away", 0.2)]),
        ("D1 identity: him", "Image 1 is a character sheet of an old man; image 2 is a shot of a film. Is the man in image 2 the same "
         "person as on the sheet (same beard, cap, coat)?", [("@sheet_ilya", 0), ("he", 0.5)]),
        ("D4 place: the same station", "Are images 1, 2 and 3 shots in the same small railway station (the same lamps, brick and green "
         "steam train)?", [("she", 0.5), ("he", 0.5), ("away", 0.2)]),
        ("D5 look: the same light", "Do images 1, 2 and 3 share one lighting and colour palette (warm golden light)? Give 1 if one of them "
         "looks cold or differently graded.", [("she", 0.5), ("he", 0.5), ("away", 0.2)]),
        ("D7 screen direction: they face each other", "In a shot and reverse shot the two people face opposite directions. In image 1 the "
         "woman should face right and in image 2 the man should face left. Does that hold?", [("she", 0.5), ("he", 0.5)]),
        ("D2 prop: the same train", "Image 1 is a reference picture of a green steam locomotive; image 2 is a shot of a film. Is the "
         "train in image 2 the same locomotive (green, brass fittings, red wheels)?", [("@train", 0), ("away", 0.3)]),
        ("D10 performance: the goodbye reads", "Images 1, 2 and 3 are the shots of a farewell. Does the emotion read as a tender goodbye "
         "(her glistening eyes, his kind nod, her alone and waving at the end)?", [("she", 0.7), ("he", 0.7), ("away", 0.8)]),
    ],
}
TROPES = {"T14": T14, "T30": T30}


def keyframe(clip, dst, frac, width=640):
    d = K.probe(clip)["seconds"]
    subprocess.run([*K.FFMPEG, "-y", "-v", "error", "-ss", f"{max(0.0, d * frac):.3f}", "-i", str(clip), "-frames:v", "1",
                    "-vf", f"scale={width}:-1", "-q:v", "3", str(dst)], check=True)
    return dst


def data_url(p):
    mime = "image/png" if str(p).endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode()


def ask(api, model, question, images):
    """The judge's score (1-5) and its reason for one question about some pictures"""
    content = []
    for i, p in enumerate(images, 1):
        content += [{"type": "text", "text": f"Image {i}:"}, {"type": "image_url", "image_url": {"url": data_url(p)}}]
    content.append({"type": "text", "text": f"{question}\n\nAnswer with a single JSON object: {{\"score\": <1-5, 5 = clearly yes, 1 = clearly "
                                            "no>, \"why\": \"<one sentence>\"}} and nothing else."})
    body = {"model": model, "temperature": 0, "max_tokens": 2000, "messages": [{"role": "user", "content": content}]}
    r = json.loads(api._req("POST", "/v1/chat/completions", body, 900))
    text = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    m = re.search(r'"score"\s*:\s*([1-5])', text)
    why = re.search(r'"why"\s*:\s*"([^"]*)"', text)
    return (int(m.group(1)) if m else None), (why.group(1) if why else text.strip()[:200])


def make(api, a, key, root):
    t = TROPES[key]
    out = root / key
    out.mkdir(parents=True, exist_ok=True)
    w, h = a.width or (384 if a.quick else 768), a.height or (288 if a.quick else 576)
    steps = a.steps or (8 if a.quick else 12)
    upscale = a.upscale if a.upscale is not None else (1.0 if a.quick else 1.5)
    run = time.strftime("%Y%m%d-%H%M%S")
    project = f"{key} {t['title']} {run}"
    report = {"trope": key, "title": t["title"], "project": project, "settings": {"width": w, "height": h, "steps": steps, "upscale": upscale,
              "seed": a.seed}, "clips": {}, "checks": {}, "judge": {}}
    files, pitches = {}, {}

    def up(name, path):
        files[Path(name).stem] = api.rpc("video.upload", {"name": f"{key.lower()}_{name}", "data": base64.b64encode(Path(path).read_bytes()).decode()})["name"]

    for m in ("minimax-h3", "minimax-h3-q6k", "minimax-h3-q4km", "minimax-h3-ref2va"):
        try:
            api.rpc("system.unload", {"model": m})
        except RuntimeError:
            pass
    for p, (prompt, size) in t["pictures"].items():
        dst = out / f"{p}.png"
        if not dst.exists():
            print(f"[{key}] picture {p}: {a.image_model} {size}", flush=True)
            r = api.rpc("image.generate", {"model": a.image_model, "prompt": prompt, "size": size, "steps": 28, "seed": a.seed})
            url = r["data"][0]["url"]
            api.get(url[url.index("/v1/"):], dst)
        up(f"{p}.png", dst)
    api.rpc("system.unload", {"model": a.image_model})
    for who in t["voices"]:
        dst = out / f"voice_{who}.wav"
        if not dst.exists():
            print(f"[{key}] voice {who}: {a.speech_model}", flush=True)
            r = api.rpc("audio.speak", {"model": a.speech_model, "input": H.VOICE[who][1], "language": "english", "voice": H.VOICE[who][0],
                                        "instructions": H.WARM, "response_format": "url"})
            url = r.get("url") or r["data"][0]["url"]
            api.get(url[url.index("/v1/"):], dst)
        up(f"voice_{who}.wav", dst)
        pitches[who] = H.pitch(dst, 16000)
    if t["voices"]:
        api.rpc("system.unload", {"model": a.speech_model})
        report["voices_hz"] = pitches
        print(f"[{key}] voices: pitch {pitches}", flush=True)

    t0, local, frames_of = time.time(), [], {}
    for i, c in enumerate(t["clips"], 1):
        if len(c) == 5:   # T14: no speech
            cid, engine, refs, secs, prompt = c
            who, line = None, None
        else:             # T30: (id, engine, refs, secs, (line, who), prompt builder)
            cid, engine, refs, secs, (line, who), build = c
            prompt = build(line)
        nf = H.frames_for(line) if line else H.snap((secs or 5.0) * FPS)
        body = {"prompt": prompt, "seconds": round(nf / FPS, 4), "steps": steps, "seed": a.seed + i, "width": w, "height": h, "project": project,
                "queue": True, "label": f"{key} {run} {i:02}/{len(t['clips']):02}: {cid}", "engine": engine,
                "ref_images": [files[r] for r in refs if not r.startswith("@")]}
        auds = [files[r[1:]] for r in refs if r.startswith("@voice")]
        if auds:
            body["ref_audios"] = auds
        if upscale > 1:
            body |= {"upscale": upscale, "upscaler": "esrgan-general"}
        t1 = time.time()
        api.rpc("video.generate", body)
        print(f"{body['label']}: queued ({nf} frames{', ' + str(len(line.split())) + ' words' if line else ''})", flush=True)
        name = K.wait_clip(api, project, i, body["label"])
        dst = api.get(f"/v1/video/files/{name}", out / f"{i:02}_{cid}.mp4")
        p = K.probe(dst)
        H.sheet(dst, out / f"{i:02}_{cid}.png")
        row = {"file": name, "frames": p["frames"], "asked": nf, "width": p["width"], "height": p["height"], "audio": p["audio"],
               "loudness_lufs": K.loudness(dst), "wall_s": round(time.time() - t1), "who": who, "pitch_hz": H.pitch(dst) if who else None}
        report["clips"][cid] = row
        print(f"{cid}: {name} in {row['wall_s']} s, {p['width']}x{p['height']}, {p['seconds']:.2f} s"
              f"{', pitch ' + str(row['pitch_hz']) + ' Hz' if who else ''}", flush=True)
        local.append((cid, dst, who))

    film = out / "film.mp4"
    total = H.cut_join([d for _, d, _ in local], [-18.0 if who else -22.0 for _, _, who in local], film)
    fp = K.probe(film)
    ck = report["checks"]
    cl = report["clips"].values()
    ck["clips are the size asked"] = all(c["width"] >= w and c["height"] >= h for c in cl)
    ck["clips are the length asked"] = all(c["frames"] == c["asked"] for c in cl)
    ck["clips have sound"] = all(c["audio"] and c["loudness_lufs"] is not None for c in cl)
    ck["film is the clips cut together"] = abs(fp["frames"] - sum(c["frames"] for c in cl)) <= 1 and fp["audio"]
    for who, ref in pitches.items():
        mine = [c["pitch_hz"] for c in cl if c["who"] == who]
        ck[f"{who}'s voice held (within 25% of the sample's {ref} Hz)"] = bool(ref) and all(m is not None and abs(m - ref) / ref <= 0.25 for m in mine)
    if len(pitches) == 2:
        (wa, wb) = list(pitches)
        a_hz = [c["pitch_hz"] for c in cl if c["who"] == wa and c["pitch_hz"]]
        b_hz = [c["pitch_hz"] for c in cl if c["who"] == wb and c["pitch_hz"]]
        ck[f"{wa}'s clips are higher than {wb}'s"] = bool(a_hz and b_hz) and min(a_hz) > max(b_hz)

    # the judge: frames of the clips and the reference pictures, one question a dimension
    if not a.no_judge:
        for m in ("minimax-h3", "minimax-h3-q6k", "minimax-h3-q4km", "minimax-h3-ref2va"):
            try:
                api.rpc("system.unload", {"model": m})
            except RuntimeError:
                pass
        clip_file = {cid: d for cid, d, _ in local}
        print(f"[{key}] judging with {a.judge_model}", flush=True)
        for dim, question, shots in t["judge"]:
            imgs = []
            for src, frac in shots:
                if src.startswith("@"):
                    imgs.append(out / f"{src[1:]}.png")
                else:
                    imgs.append(keyframe(clip_file[src], out / f"frame_{src}_{int(frac * 100):02}.jpg", frac))
            score, why = ask(api, a.judge_model, question, imgs)
            report["judge"][dim] = {"score": score, "why": why, "frames": [p.name for p in imgs]}
            ck[f"{dim} (judge {score}/5, 4 passes)"] = score is not None and score >= 4
            print(f"  {dim}: {score}/5 - {why}", flush=True)
    report["film"] = {"file": str(film), "seconds": fp["seconds"], "wall_s": round(time.time() - t0)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    ok = True
    for k, v in ck.items():
        print(f"{'ok  ' if v else 'FAIL'} [{key}] {k}")
        ok &= bool(v)
    print(f"[{key}] film: {film}, {fp['width']}x{fp['height']}, {fp['seconds']:.2f} s; report {out / 'report.json'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token")
    ap.add_argument("--out", default="trope-scenes")
    ap.add_argument("--only", default="T14,T30", help="the tropes to make (T14, T30)")
    ap.add_argument("--quick", action="store_true", help="384x288, 8 steps, no upscale")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--upscale", type=float)
    ap.add_argument("--seed", type=int, default=1954)
    ap.add_argument("--image-model", default="qwen-image-2.1-q8")
    ap.add_argument("--speech-model", default="qwen3-tts-custom")
    ap.add_argument("--judge-model", default="qwen3.8-27b-q4")
    ap.add_argument("--no-judge", action="store_true", help="the measured checks and the sheets only")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    a = ap.parse_args()
    K.FFMPEG, K.FFPROBE = shlex.split(a.ffmpeg), shlex.split(a.ffprobe)
    K.ffmpeg_can()
    keys = [k.strip().upper() for k in a.only.split(",") if k.strip()]
    for k in keys:
        if k not in TROPES:
            sys.exit(f"{k}: the tropes here are {', '.join(TROPES)}")
    api = K.Api(a.url, K.token_of(a.token))
    root = Path(a.out)
    ok = True
    for k in keys:
        ok &= make(api, a, k, root)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
