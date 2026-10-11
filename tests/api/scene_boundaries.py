#!/usr/bin/env python3
"""Reference test: every kind of scene boundary, made each way the model offers, through the front door's API.

A film is scenes, and what matters at a cut is what must stay the same across it: the motion, the face, the voice,
the look, the place, or nothing. This makes one short clip A, then for each kind of boundary the clip B that follows
it by each candidate recipe (the keys that carry something across: first_frame / last_frame, guide_clip +
first_audio, ref_images / ref_videos / ref_audios of Ref2VA, effect embeddings, a control video, a regenerated
region) - and measures and pictures the result, so the best recipe for a kind of scene is read off, not guessed.
docs/video/scenes.md is the table it settled.

    tests/api/scene_boundaries.py --url http://box:8000                # everything at 384x384, ~1 h
    tests/api/scene_boundaries.py --only S1,S3                         # some boundaries (their clip A is made too)
    tests/api/scene_boundaries.py --list                               # the boundaries and their recipes

Each recipe is one clip. Writes into --out: the assets, each clip, a contact sheet per clip (frames across the clip,
so a face that changes or a motion that restarts is seen) and report.json (the numbers: the seam, the voice's
spectral centroid, how far a regenerated clip left the rest alone). Clips are listed in one project, grouped by
engine so the daemon swaps its denoiser once. Exits 1 when a clip fails.

Needs: Python 3.9+, an ffmpeg and ffprobe with H.264; on the server minimax-h3 (INT8) and minimax-h3-ref2va, an
image model (qwen-image-2.1-q8) and a speech model (qwen3-tts-custom). The token: --token, else serve.json here.
"""

import argparse
import base64
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaiju_lighthouse as K   # the API client, the probes, the polling

FPS = 24
STYLE = "Live-action, cinematic, natural light, shallow depth of field."
MARA = "a woman in her forties with short grey-streaked black hair, grey eyes and a weathered face, in a navy wool coat with silver buttons"
ILYA = "an old man with a full white beard, a flat cap and oil-stained green overalls"

PICTURES = {   # name -> image.generate prompt
    "mara": f"portrait photograph, waist-up, {MARA}, standing on a harbour quay in soft overcast light, looking at the camera, cinematic",
    "ilya": f"portrait photograph, waist-up, {ILYA}, in a ship's engine room with brass pipes, warm light, looking at the camera, cinematic",
    "dock": f"wide photograph of a long wooden harbour dock at dusk with fishing boats, {MARA}, seen small from behind in the middle distance, cinematic",
    "market": "wide photograph of an empty cobbled fish market street at morning, awnings and crates, no people, cinematic",
    "clock": "photograph of a large brass ship's clock face on a wall, close-up, warm light, cinematic",
}
VOICES = {   # name -> (voice, the line)
    "mara": ("vivian", "We sail at first light, and the tide will not wait for any of us."),
    "ilya": ("ryan", "The engine will hold, captain, but only if you ask it nicely."),
}


def shot(action, sound, music="None."):
    return f"integrated_multimodal_description: [Shot 1] {STYLE} {action}\n\noverall_soundscape: {sound}\n\nnon_diegetic_music: {music}"


def say(who, line):
    """A speaker's id and line, in the model's dialogue tag"""
    desc = "a warm low alto" if who == "mara" else "a gravelly baritone"
    return f"says in {desc} voice (S1): <d>[English] {line}</d>"


# Recipes: (id, scene kind, what the boundary asks, engine, what it needs made first, the request's keys, a prompt).
# Values in {braces} are made files; "{a_run}" is the clip A of that name and "{a_run.last}" its last frame (PNG).
# The request always has the draft's size, steps and seed; "frames" is a length in frames (17k + 5) as the clip needs.
R = []


def recipe(id, kind, asks, engine, needs, body, prompt, frames=107, **kw):
    R.append({"id": id, "kind": kind, "asks": asks, "engine": engine, "needs": needs, "body": body, "prompt": prompt,
              "frames": frames, **kw})


# ---- clips A: the scenes the boundaries follow
recipe("a_run", "A", "a scene to continue: the captain runs along the dock (from a picture)", "INT8", ["dock"],
       {"first_frame": "{dock}"},
       shot("The woman in the navy coat runs along the wooden dock away from the camera toward the boats, her coat flapping, "
            "boots pounding the planks. The camera tracks behind her at a steady distance.", "Boots pounding wooden planks, a rope creaking, gulls."))
recipe("a_talk", "A", "a scene to continue: the captain speaks to camera (from a picture)", "INT8", ["mara"],
       {"first_frame": "{mara}"},
       shot(f"A medium close-up of {MARA}, who blinks and looks straight into the camera, then {say('mara', VOICES['mara'][1])} "
            "The camera sits in one fixed position on a tripod; that exact shot size holds from the first frame to the last.",
            "Her clear voice, a faint harbour hush."))

# ---- S1: the same shot goes on (motion and sound must carry)
CONT = ("Continuing the shot, the woman in the navy coat keeps running along the wooden dock away from the camera, then "
        "slows at the end of the dock and turns to look back over her shoulder. The camera tracks behind her at a steady distance.")
CONT_SND = "Boots pounding wooden planks slowing to a stop, a rope creaking, gulls."
recipe("s1_guide", "S1 the shot goes on", "motion + sound: the last 22 frames and their sound", "INT8", ["a_run"],
       {"guide_clip": "{a_run}:22:0", "first_audio": "{a_run}", "first_audio_s": 22 / FPS}, shot(CONT, CONT_SND), frames=124)
recipe("s1_still", "S1 the shot goes on", "the last frame only (a still)", "INT8", ["a_run"],
       {"first_frame": "{a_run}"}, shot(CONT, CONT_SND), frames=107)
recipe("s1_refclip", "S1 the shot goes on", "the clip as <Video 1> (its motion and sound as a reference) + its last frame as <Picture 1>",
       "REF2VA", ["a_run", "a_run.last"], {"ref_videos": ["{a_run}"], "ref_images": ["{a_run.last}"]},
       shot("The woman in the navy coat from <Picture 1> keeps running along the wooden dock away from the camera as in <Video 1>, "
            "then slows at the end of the dock and turns to look back over her shoulder. The camera tracks behind her at a steady distance.",
            CONT_SND), frames=107)

# ---- S2: the same person speaks again (face and voice must carry, the shot stays)
LINE2 = "Ilya, bring her up to full steam and tell the crew I am coming aboard."
S2 = shot(f"A medium close-up of {MARA}, who blinks and looks straight into the camera, then {say('mara', LINE2)} "
          "The camera sits in one fixed position on a tripod; that exact shot size holds from the first frame to the last.",
          "Her clear voice, a faint harbour hush.")
recipe("s2_hub", "S2 the same speaker, the next line", "the picture at both ends (a hub), no sound carried", "INT8", ["mara"],
       {"first_frame": "{mara}", "last_frame": "{mara}"}, S2)
recipe("s2_guide", "S2 the same speaker, the next line", "motion + sound from clip A", "INT8", ["a_talk"],
       {"guide_clip": "{a_talk}:22:0", "first_audio": "{a_talk}", "first_audio_s": 22 / FPS}, S2, frames=124)
recipe("s2_ref", "S2 the same speaker, the next line", "her picture and her voice as references (no frame pinned)", "REF2VA", ["mara", "voice_mara"],
       {"ref_images": ["{mara}"], "ref_audios": ["{voice_mara}"]},
       shot(f"A medium close-up of the woman from <Picture 1>, who blinks and looks straight into the camera, then speaks in the voice of "
            f"<Audio 1> (S1): <d>[English] {LINE2}</d> The camera sits in one fixed position on a tripod; that exact shot size holds "
            "from the first frame to the last.", "Her clear voice, a faint harbour hush."))
recipe("s2_ref_pin", "S2 the same speaker, the next line", "references + the picture pinned as first and last frame", "REF2VA", ["mara", "voice_mara"],
       {"ref_images": ["{mara}"], "ref_audios": ["{voice_mara}"], "first_frame": "{mara}", "last_frame": "{mara}"},
       shot(f"A medium close-up of the woman from <Picture 1>, who blinks and looks straight into the camera, then speaks in the voice of "
            f"<Audio 1> (S1): <d>[English] {LINE2}</d> The camera sits in one fixed position on a tripod; that exact shot size holds "
            "from the first frame to the last.", "Her clear voice, a faint harbour hush."))

# ---- S3: the speaker changes (the reverse shot: another face, another voice)
LINE3 = VOICES["ilya"][1]
S3A = shot(f"A medium close-up of {ILYA}, who blinks and looks straight into the camera, then {say('ilya', LINE3)} "
           "The camera sits in one fixed position on a tripod; that exact shot size holds from the first frame to the last.",
           "His low voice, a faint engine hum.")
recipe("s3_pic", "S3 the speaker changes", "his picture as the first frame (no voice held)", "INT8", ["ilya"], {"first_frame": "{ilya}"}, S3A)
recipe("s3_ref", "S3 the speaker changes", "his picture and voice as references", "REF2VA", ["ilya", "voice_ilya"],
       {"ref_images": ["{ilya}"], "ref_audios": ["{voice_ilya}"]},
       shot(f"A medium close-up of the old man from <Picture 1>, who blinks and looks straight into the camera, then speaks in the voice of "
            f"<Audio 1> (S1): <d>[English] {LINE3}</d> The camera sits in one fixed position on a tripod; that exact shot size holds "
            "from the first frame to the last.", "His low voice, a faint engine hum."))
recipe("s3_two", "S3 the speaker changes", "both people in one shot: two pictures, two voices", "REF2VA",
       ["mara", "ilya", "voice_mara", "voice_ilya"],
       {"ref_images": ["{mara}", "{ilya}"], "ref_audios": ["{voice_mara}", "{voice_ilya}"]},
       shot("A medium two-shot: the woman from <Picture 1> on the left and the old man from <Picture 2> on the right stand side by side "
            "in a ship's engine room, both facing the camera. The woman speaks in the voice of <Audio 1> (S1): <d>[English] Will she hold?</d> "
            "Then the old man answers in the voice of <Audio 2> (S2): <d>[English] She will.</d> The camera is static.",
            "Their two voices, a faint engine hum."), frames=124)

# ---- S4: a hard cut to the same person somewhere new (identity carries, the place does not)
S4 = shot("A wide shot: the woman walks through a cobbled fish market in the morning, past crates and awnings, and stops at a stall. "
          "The camera tracks beside her at a steady distance.", "Footsteps on cobbles, distant stalls, a gull.")
recipe("s4_place", "S4 the same person, a new place", "the new place's picture only (the person is invented)", "INT8", ["market"],
       {"first_frame": "{market}"}, S4)
recipe("s4_person", "S4 the same person, a new place", "her picture as <Picture 1>, the place in the prompt", "REF2VA", ["mara"],
       {"ref_images": ["{mara}"]},
       shot("A wide shot: the woman from <Picture 1> walks through a cobbled fish market in the morning, past crates and awnings, and stops "
            "at a stall. The camera tracks beside her at a steady distance.", "Footsteps on cobbles, distant stalls, a gull."))
recipe("s4_both", "S4 the same person, a new place", "her picture <Picture 1> and the place's <Picture 2>", "REF2VA", ["mara", "market"],
       {"ref_images": ["{mara}", "{market}"]},
       shot("A wide shot: the woman from <Picture 1> walks through the cobbled fish market of <Picture 2>, past crates and awnings, and "
            "stops at a stall. The camera tracks beside her at a steady distance.", "Footsteps on cobbles, distant stalls, a gull."))

# ---- S5: a new scene in the same look (an establishing shot: nothing but the look carries)
S5 = shot("A wide establishing shot of a harbour town at dusk: grey stone houses climb the hill above a still bay, fishing boats at their "
          "moorings, a thin mist on the water. The camera is static.", "Water lapping, a distant gull, a bell far off.")
recipe("s5_plain", "S5 a new scene, the same look", "words alone", "INT8", [], {}, S5)
recipe("s5_look", "S5 a new scene, the same look", "clip A's last frame as <Picture 1>, named as the look", "REF2VA", ["a_run.last"],
       {"ref_images": ["{a_run.last}"]},
       shot("A wide establishing shot of a harbour town at dusk in the exact look and light of <Picture 1>: grey stone houses climb the hill "
            "above a still bay, fishing boats at their moorings, a thin mist on the water. The camera is static.",
            "Water lapping, a distant gull, a bell far off."))

# ---- S6: a match cut / a transformation (A's last frame becomes something else)
S6 = shot("The camera pushes in slowly on the weathered wooden dock, and the planks and sky dissolve and reshape themselves until the whole "
          "frame is the face of a large brass ship's clock, its hands ticking.", "A slow ticking, a low swell of wind.")
recipe("s6_fl", "S6 a match cut / transformation", "A's last frame to a target picture (first_frame + last_frame)", "INT8", ["a_run", "clock"],
       {"first_frame": "{a_run}", "last_frame": "{clock}"}, S6)
recipe("s6_guide_fl", "S6 a match cut / transformation", "A's motion guide + a target picture as the last frame", "INT8", ["a_run", "clock"],
       {"guide_clip": "{a_run}:22:0", "last_frame": "{clock}", "first_audio": "{a_run}", "first_audio_s": 22 / FPS}, S6, frames=124)

# ---- S7: an effect beat inside a scene (an effect embedding, the scene carried)
recipe("s7_effect", "S7 an effect beat", "bullet time on A's motion guide", "INT8", ["a_run"],
       {"guide_clip": "{a_run}:22:0", "first_audio": "{a_run}", "first_audio_s": 22 / FPS},
       shot("Continuing the shot, the woman in the navy coat leaps off the end of the dock; embedding:minimaxh3_bullet_time the camera "
            "circles her in frozen time as spray hangs in the air around her, then time resumes and she plunges into the water. "
            "The camera is static before the leap.", "Boots on planks, then a deep hush, then a huge splash."), frames=124)

# ---- S8: a loop
recipe("s8_loop", "S8 a seamless loop", "the same picture as first and last frame", "INT8", ["mara"],
       {"first_frame": "{mara}", "last_frame": "{mara}"},
       shot(f"A medium close-up of {MARA}, who blinks slowly and breathes, a breeze moving her hair and coat collar, light waves behind her. "
            "The camera sits in one fixed position on a tripod; that exact shot size and light hold from the first frame to the last.",
            "Wind and water, softly."))

# ---- S9: another person does what A did (motion transfer: identity from a picture, motion from a clip)
recipe("s9_move", "S9 motion transferred", "his picture as <Picture 1>, A's running as <Video 1>", "REF2VA", ["ilya", "a_run"],
       {"ref_images": ["{ilya}"], "ref_videos": ["{a_run}"], "ref_video_sound": False},
       shot("The old man from <Picture 1> runs along a wooden harbour dock away from the camera, his cap on, moving exactly as in <Video 1>. "
            "The camera tracks behind him at a steady distance.", "Boots pounding wooden planks, a rope creaking, gulls."))

recipe("s9_desc", "S9 motion transferred", "the same, the man described in words as well", "REF2VA", ["ilya", "a_run"],
       {"ref_images": ["{ilya}"], "ref_videos": ["{a_run}"], "ref_video_sound": False},
       shot("Using the movement of the figure in <Video 1> only, the white-bearded old man in a flat cap and oil-stained green overalls "
            "from <Picture 1> runs along a wooden harbour dock away from the camera. Every person on screen is that old man. "
            "The camera tracks behind him at a steady distance.", "Boots pounding wooden planks, a rope creaking, gulls."))
recipe("s9_pic", "S9 motion transferred", "his picture alone, the running in words (the baseline)", "REF2VA", ["ilya"],
       {"ref_images": ["{ilya}"]},
       shot("The white-bearded old man in a flat cap and oil-stained green overalls from <Picture 1> runs along a wooden harbour dock "
            "away from the camera. The camera tracks behind him at a steady distance.", "Boots pounding wooden planks, a rope creaking, gulls."))
recipe("s9_control", "S9 motion transferred", "A's edges as a control video, his picture as the first frame", "INT8", ["ilya", "a_run.canny"],
       {"control_video": "{a_run.canny}", "control_strength": 0.7, "first_frame": "{ilya}"},
       shot("The white-bearded old man in a flat cap and oil-stained green overalls runs along a wooden harbour dock away from the "
            "camera. The camera tracks behind him at a steady distance.", "Boots pounding wooden planks, a rope creaking, gulls."))

# ---- S10: a voice over another scene (b-roll under her voice)
recipe("s10_vo", "S10 voice-over b-roll", "her voice as <Audio 1> over the market, lips closed", "REF2VA", ["market", "mara", "voice_mara"],
       {"ref_images": ["{market}"], "ref_audios": ["{voice_mara}"]},
       shot("A slow wide shot of the empty cobbled fish market at morning, awnings stirring in the breeze, crates stacked and wet cobbles "
            "shining. The woman from the earlier scene speaks in an off-screen voiceover in the voice of <Audio 1>: <d>[English] We sail "
            "at first light, and the tide will not wait for any of us.</d> while no one is on screen. The camera is static.",
            "Her voice-over, distant gulls, water on stone."))

# ---- S11: the same composition, another style (a control video: canny edges of A)
recipe("s11_canny", "S11 the same motion, another style", "A's edges as a control video", "INT8", ["a_run.canny"],
       {"control_video": "{a_run.canny}", "control_strength": 1.0},
       shot("A hand-painted watercolour animation: a woman in a navy coat runs along a wooden dock away from the camera toward fishing "
            "boats, soft washes of blue and ochre, visible paper grain. The camera tracks behind her at a steady distance.",
            "Boots pounding wooden planks, a rope creaking, gulls."))

# ---- S12: a fix inside a scene (regenerate a region of A, the rest kept)
recipe("s12_regen", "S12 a fix inside a scene", "A with a box (x0,y0,x1,y1) regenerated: the coat made red", "INT8", ["a_run"],
       {"source": "{a_run}", "regen": "0-4.4", "regen_box": "60,40,330,330"},
       shot("The woman in a bright red coat runs along the wooden dock away from the camera toward the boats, her coat flapping, boots "
            "pounding the planks. The camera tracks behind her at a steady distance.", "Boots pounding wooden planks, a rope creaking, gulls."))
recipe("s12_tail", "S12 a fix inside a scene", "A's first 2 s kept, the rest regenerated (another ending)", "INT8", ["a_run"],
       {"source": "{a_run}", "regen": "2.0-4.4"},
       shot("The woman in the navy coat runs along the wooden dock away from the camera toward the boats, then stops, drops to her "
            "knees and puts both hands over her face. The camera tracks behind her at a steady distance.",
            "Boots pounding wooden planks, then silence, a gull."))


def lufs_centroid(f):
    """The sound's mean spectral centroid (Hz): a crude timbre, enough to tell a voice that held from one that did not"""
    o = subprocess.run([*K.FFMPEG, "-v", "error", "-i", str(f), "-vn", "-af",
                        "aspectralstats=measure=centroid,ametadata=print:key=lavfi.aspectralstats.1.centroid:file=-", "-f", "null", "-"],
                       capture_output=True, text=True).stdout
    v = [float(m) for m in re.findall(r"centroid=([0-9.]+)", o)]
    v = [x for x in v if x > 0]
    return round(sum(v) / len(v)) if v else None


def sheet(f, dst, n=6, size=160):
    """n frames across the clip as one picture"""
    subprocess.run([*K.FFMPEG, "-y", "-v", "error", "-i", str(f), "-vf",
                    f"select='not(mod(n\\,{max(1, K.probe(f)['frames'] // n)}))',scale={size}:-1,tile={n}x1", "-frames:v", "1", str(dst)], check=True)


def seam(a, b, at=0):
    """How like clip A's last frame B's frame `at` is (grey correlation: 1 the same picture): a guide's last frame is 21"""
    return round(K.corr(K.thumb(a, last=True), K.thumb(b, at=at)), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token")
    ap.add_argument("--out", default="scene-boundaries")
    ap.add_argument("--only", default="", help="boundaries (S1,S3 ...) or recipe ids, comma-separated; what they need is made too")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1954)
    ap.add_argument("--image-model", default="qwen-image-2.1-q8")
    ap.add_argument("--speech-model", default="qwen3-tts-custom")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    a = ap.parse_args()
    if a.list:
        for r in R:
            print(f"{r['id']:12} {r['engine']:6} {r['kind']:36} {r['asks']}")
        return 0
    K.FFMPEG, K.FFPROBE = shlex.split(a.ffmpeg), shlex.split(a.ffprobe)
    K.ffmpeg_can()
    want = {w.strip().lower() for w in a.only.split(",") if w.strip()}
    pick = [r for r in R if r["kind"] == "A" or not want or r["id"] in want or r["kind"].split()[0].lower() in want]
    todo = {r["id"] for r in pick if r["kind"] != "A"}
    # clips A only if something picked needs them (or they are asked for by id)
    need = {n.split(".")[0] for r in pick if r["id"] in todo for n in r["needs"]}
    pick = [r for r in pick if r["kind"] != "A" or r["id"] in need or r["id"] in want]

    api = K.Api(a.url, K.token_of(a.token))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = time.strftime("%Y%m%d-%H%M%S")
    project = f"scene-boundaries {run}"
    size = a.size
    files = {}      # a name -> the file in the studio's output
    state_f = out / "state.json"   # clips made in an earlier run (the same size and steps) are reused
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    state = state if state.get("size") == [size, a.steps, a.seed] else {"size": [size, a.steps, a.seed], "clips": {}}
    report = {"project": project, "size": size, "steps": a.steps, "clips": {}}

    def up(name, data):
        files[Path(name).stem] = api.rpc("video.upload", {"name": name, "data": base64.b64encode(data).decode()})["name"]

    # 1. assets: pictures, voices, the control video
    need_all = {n.split(".")[0] for r in pick for n in r["needs"]} | {n for r in pick for n in r["needs"]}
    pics = [p for p in PICTURES if p in need_all]
    if pics or any(n.startswith("voice_") for n in need_all):
        for m in ("minimax-h3", "minimax-h3-q6k", "minimax-h3-q4km", "minimax-h3-ref2va"):   # the card for the image and speech models
            try:
                api.rpc("system.unload", {"model": m})
            except RuntimeError:
                pass
    if pics:
        for p in pics:
            dst = out / f"{p}.png"
            if not dst.exists():
                print(f"picture {p}: {a.image_model}", flush=True)
                w, h = (size, size)
                r = api.rpc("image.generate", {"model": a.image_model, "prompt": PICTURES[p], "size": f"{max(512, w)}x{max(512, h)}",
                                               "steps": 20, "seed": a.seed})
                url = r["data"][0]["url"]
                api.get(url[url.index("/v1/"):], dst)
            up(f"{p}.png", dst.read_bytes())
        api.rpc("system.unload", {"model": a.image_model})
    for v in [n for n in need_all if n.startswith("voice_")]:
        who = v[6:]
        dst = out / f"{v}.wav"
        if not dst.exists():
            print(f"voice {who}: {a.speech_model}", flush=True)
            r = api.rpc("audio.speak", {"model": a.speech_model, "input": VOICES[who][1], "voice": VOICES[who][0], "language": "english",
                                        "response_format": "url"})
            url = r.get("url") or r["data"][0]["url"]
            api.get(url[url.index("/v1/"):], dst)
        up(f"{v}.wav", dst.read_bytes())
    if pics or any(n.startswith("voice_") for n in need_all):
        api.rpc("system.unload", {"model": a.speech_model})

    # 2. the clips, one engine at a time, in the order given (A's before the boundaries that follow them)
    order = {"INT8": 0, "REF2VA": 1}
    done_ids = set()
    for r in sorted(pick, key=lambda r: (order[r["engine"]], r["kind"] != "A")):
        for n in r["needs"]:
            if n.endswith(".last"):
                base = n[:-5]
                if f"{n}" not in files and base in files:
                    files[n] = files[base].replace(".mp4", ".last.png")
            if n.endswith(".canny") and n not in files:
                src = out / f"{n.split('.')[0]}.mp4"
                dst = out / f"{n.split('.')[0]}.canny.mp4"
                subprocess.run([*K.FFMPEG, "-y", "-v", "error", "-i", str(src), "-vf", "edgedetect=low=0.1:high=0.3,format=yuv420p", "-an",
                                "-c:v", "libx264", "-crf", "16", str(dst)], check=True)
                up(dst.name, dst.read_bytes())
        body = {"prompt": r["prompt"], "seconds": round(r["frames"] / FPS, 4), "steps": a.steps, "seed": a.seed, "width": size, "height": size,
                "engine": r["engine"], "project": project, "queue": True,
                "label": f"Atlas {run} {len(report['clips']) + 1:02}/{len(pick):02}: {r['id']} - {r['asks']}"}
        if any(n.split(".")[0] not in files and n not in files for n in r["needs"]):
            print(f"{r['id']}: skipped (a clip it needs failed)", flush=True)
            report["clips"][r["id"]] = {"label": r["id"], "error": "a clip it needs failed"}
            continue
        for k, v in r["body"].items():
            def sub(s):
                if isinstance(s, str):
                    m = re.fullmatch(r"\{([a-z0-9_.]+)\}(:.*)?", s)
                    if m:
                        return files[m.group(1)] + (m.group(2) or "")
                return s
            body[k] = [sub(x) for x in v] if isinstance(v, list) else sub(v)
        label = body["label"]
        old = state["clips"].get(r["id"])
        if old and (out / f"{r['id']}.mp4").exists() and not want & {r["id"], r["kind"].split()[0].lower()}:
            print(f"{r['id']}: reusing {old['file']}", flush=True)
            files[r["id"]] = old["file"]
            report["clips"][r["id"]] = old
            continue
        n = len(report["clips"]) + 1
        t = time.time()
        try:
            api.rpc("video.generate", body)
            print(f"{label}: queued", flush=True)
            name = K.wait_clip(api, project, n, label)
        except RuntimeError as e:
            print(f"{label}: FAILED {e}", flush=True)
            report["clips"][r["id"]] = {"label": label, "error": str(e)}
            try:
                api.rpc("video.queue.resume")
            except RuntimeError:
                pass
            continue
        dst = api.get(f"/v1/video/files/{name}", out / f"{r['id']}.mp4")
        files[r["id"]] = name
        p = K.probe(dst)
        row = {"label": label, "file": name, "engine": r["engine"], "wall_s": round(time.time() - t), **p}
        sheet(dst, out / f"{r['id']}.png")
        if r["kind"] != "A" and "a_run" in r["needs"] + [r["body"].get("source", "")] and (out / "a_run.mp4").exists():
            row["seam_with_a_run"] = seam(out / "a_run.mp4", dst, 21 if "guide_clip" in r["body"] else 0)
        if r["kind"] != "A" and "a_talk" in r["needs"] and (out / "a_talk.mp4").exists():
            row["seam_with_a_talk"] = seam(out / "a_talk.mp4", dst, 21 if "guide_clip" in r["body"] else 0)
        if p["audio"]:
            row["centroid_hz"] = lufs_centroid(dst)
        report["clips"][r["id"]] = row
        state["clips"][r["id"]] = row
        state_f.write_text(json.dumps(state))
        print(f"{label}: {name} in {row['wall_s']} s, {p['width']}x{p['height']}, {p['seconds']:.2f} s"
              f"{', seam ' + str(row.get('seam_with_a_run', row.get('seam_with_a_talk', ''))) if 'seam_with_a_run' in row or 'seam_with_a_talk' in row else ''}"
              f"{', centroid ' + str(row.get('centroid_hz')) + ' Hz' if row.get('centroid_hz') else ''}", flush=True)
    for v in [n for n in need_all if n.startswith("voice_")]:
        report.setdefault("voices_hz", {})[v] = lufs_centroid(out / f"{v}.wav")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    bad = [k for k, v in report["clips"].items() if "error" in v]
    print(f"report {out / 'report.json'}; {len(report['clips']) - len(bad)} clips, {len(bad)} failed{': ' + ', '.join(bad) if bad else ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
